"""Lifecycle tests use only mock sleepers and a stdlib socket, never training/TB."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('background_job', HERE / 'background_job.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)

MOCK = '''import argparse,signal,socket,time
p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--output');p.add_argument('--num-envs');p.add_argument('--iterations');p.add_argument('--seed');p.add_argument('--num-mini-batches');p.add_argument('--identity');p.add_argument('--logdir');p.add_argument('--port',type=int)
a=p.parse_args()
if a.mode=='server':
 s=socket.socket();s.bind(('127.0.0.1',a.port));s.listen(1)
signal.signal(signal.SIGINT,lambda *unused:exit(0))
signal.signal(signal.SIGTERM,lambda *unused:exit(0))
print('MOCK_ONLY_NO_TRAINING_NO_TENSORBOARD',flush=True)
while True:time.sleep(.05)
'''


class MockController(job.Controller):
    def check_dependencies(self):
        assert self.wrapper.is_file() and self.train_script.is_file()

    def tensorboard_command(self, port):
        return [sys.executable, '-I', '-B', str(self.train_script), 'server',
                '--identity', 'tensorboard.main', '--logdir', str(self.runs), '--port', str(port)]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        test_area = job.ROOT / 'tests'
        test_area.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='launcher_fixture_', dir=str(test_area))
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/train_residual.py').write_text(MOCK)
        # Wrapper follows the same contract as the real env launcher.
        (self.root / 'launch_python.sh').write_text('#!/usr/bin/env bash\nexec ' + sys.executable + ' -I -B "$@"\n')
        self.controller = MockController(self.root)
        self.port = free_port()

    def test_configured_interpreter_is_shared_with_tensorboard(self):
        with patch.dict(os.environ,{'RESMIMIC_PYTHON':sys.executable}):
            controller=job.Controller(self.root)
            self.assertEqual(controller.tensorboard_python,Path(sys.executable).absolute())
            command=controller.tensorboard_command(self.port)
            self.assertEqual(command[0],str(Path(sys.executable).absolute()))

    def tearDown(self):
        try:
            self.controller.execute(job.parse_args(['stop-all']))
            for process in self.controller.handles:
                process.wait(timeout=2)
        finally:
            self.temp.cleanup()

    def args(self, command='start', **kwargs):
        argv = [command, '--port', str(self.port), '--num-envs', '4', '--iterations', '2']
        for key, value in kwargs.items():
            argv += ['--' + key.replace('_', '-'), str(value)]
        return job.parse_args(argv)

    def test_start_status_stop_reuse_and_private_session(self):
        first = self.controller.execute(self.args())
        self.assertEqual(first['training'], 'running')
        self.assertEqual(first['tensorboard'], 'running')
        self.assertEqual(os.getsid(first['training_pid']), first['training_pid'])
        self.assertEqual(os.getsid(first['tensorboard_pid']), first['tensorboard_pid'])
        self.assertEqual(os.readlink('/proc/%d/fd/0' % first['training_pid']), '/dev/null')
        for pid in [first['training_pid'], first['tensorboard_pid']]:
            environment = dict(item.split('=', 1) for item in Path('/proc/%d/environ' % pid).read_bytes().decode().rstrip('\0').split('\0'))
            self.assertEqual(environment['TMPDIR'], str(self.root / 'tmp'))
            self.assertEqual(environment['XDG_CACHE_HOME'], str(self.root / 'cache'))
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(Path(first['training_log']).parent, Path(first['run_dir']))
        self.assertEqual((self.root / 'state/training_latest.log').resolve(), Path(first['training_log']))
        with self.assertRaisesRegex(RuntimeError, 'already running'):
            self.controller.execute(self.args())
        stopped = self.controller.execute(self.args('stop'))
        self.assertEqual(stopped['training'], 'not running')
        self.assertEqual(stopped['tensorboard'], 'running')
        second = self.controller.execute(self.args())
        self.assertNotEqual(first['run_dir'], second['run_dir'])
        self.assertEqual(first['tensorboard_pid'], second['tensorboard_pid'])
        stopped_all = self.controller.execute(self.args('stop-all'))
        self.assertEqual(stopped_all['training'], 'not running')
        self.assertEqual(stopped_all['tensorboard'], 'not running')

    def test_background_children_survive_controller_process_exit(self):
        starter = self.root / 'starter.py'
        starter.write_text('import sys,json\nfrom pathlib import Path\nsys.path.insert(0,' + repr(str(HERE)) + ')\nfrom test_background_job import MockController,job\nprint(json.dumps(MockController(Path(' + repr(str(self.root)) + ')).execute(job.parse_args(' + repr(['start', '--port', str(self.port), '--num-envs', '4', '--iterations', '2']) + '))))\n')
        completed = subprocess.run([sys.executable, '-I', '-B', str(starter)], check=True,
                                   capture_output=True, text=True, timeout=10)
        first = json.loads(completed.stdout)
        after_parent_exit = self.controller.execute(self.args('status'))
        self.assertEqual(after_parent_exit['training'], 'running')
        self.assertEqual(after_parent_exit['tensorboard'], 'running')
        self.assertEqual(first['training_pid'], after_parent_exit['training_pid'])

    def test_port_conflict_never_kills_other_listener(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', self.port));listener.listen(1)
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                self.controller.execute(self.args())
            self.assertEqual(listener.getsockname()[1], self.port)
            self.assertFalse(self.controller.handles)

    def test_pid_reuse_and_wrong_command_are_not_owned(self):
        record = job.process_identity(os.getpid())
        record.pop('argv');record['expected_tokens'] = ['not-this-command']
        self.assertFalse(self.controller.stop_record(record))
        record['expected_tokens'] = job.process_identity(os.getpid())['argv'][-1:]
        record['start_ticks'] = str(int(record['start_ticks']) + 1)
        self.assertFalse(job.owns_process(record))
        self.assertFalse(self.controller.stop_record(record))

    def test_child_start_failure_cleans_new_tensorboard(self):
        self.controller.train_script.write_text(MOCK.replace("while True:time.sleep(.05)", "\nif a.mode=='train':exit(23)\nwhile True:time.sleep(.05)"))
        with self.assertRaisesRegex(RuntimeError, 'startup|identity'):
            self.controller.execute(self.args())
        self.assertTrue(self.controller.handles)
        for process in self.controller.handles:
            process.wait(timeout=2)
            self.assertFalse(job.process_identity(process.pid))


if __name__ == '__main__':
    unittest.main(verbosity=2)
