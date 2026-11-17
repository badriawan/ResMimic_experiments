"""Start/inspect/stop detached residual training and its local TensorBoard.

This controller imports only the standard library. Training starts only with
the start command; status, stop and --help do not import Isaac Gym or Torch.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,MOTION,OBJECT,CHECKPOINT,python_executable


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='start', choices=['start', 'status', 'stop', 'stop-all'])
    parser.add_argument('--num-envs', type=positive, default=32)
    parser.add_argument('--iterations', type=positive, default=5000)
    parser.add_argument('--num-mini-batches', type=positive)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--port', type=positive, default=6006)
    args = parser.parse_args(argv)
    if args.port > 65535:
        parser.error('--port must be <=65535')
    if args.num_mini_batches and args.num_envs * 24 % args.num_mini_batches:
        parser.error('--num-mini-batches must divide num-envs * 24')
    return args


def process_identity(pid):
    """Identify a Linux process independently of PID reuse, excluding zombies."""
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    argv=(proc / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0'))
    except (OSError, ValueError, IndexError):
        return None


def owns_process(record):
    if not isinstance(record, dict):
        return False
    actual = process_identity(record.get('pid', -1))
    return bool(actual and actual['start_ticks'] == record.get('start_ticks')
                and actual['boot_id'] == record.get('boot_id')
                and all(token in actual['argv'] for token in record.get('expected_tokens', []))
                and record.get('expected_tokens'))


def owns_listening_port(pid, port):
    try:
        sockets = {os.readlink(str(fd)) for fd in (Path('/proc') / str(pid) / 'fd').iterdir()}
        for name in ['tcp', 'tcp6']:
            for row in (Path('/proc/net') / name).read_text().splitlines()[1:]:
                fields = row.split()
                if fields[3] == '0A' and int(fields[1].split(':')[1], 16) == port and 'socket:[%s]' % fields[9] in sockets:
                    return True
    except (OSError, IndexError, ValueError):
        pass
    return False


class Controller:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        if self.root != ROOT and ROOT not in self.root.parents:
            raise ValueError('Controller files must stay inside after_tuning')
        self.state_dir = self.root / 'state'
        self.state_path = self.state_dir / 'active.json'
        self.runs = self.root / 'runs'
        self.train_script = self.root / 'scripts/train_residual.py'
        self.wrapper = self.root / 'launch_python.sh'
        self.tensorboard_python = python_executable()
        self.handles = []

    def read(self):
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text())

    def write(self, state):
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(self.state_path)

    def check_dependencies(self):
        files = [self.wrapper, self.train_script, self.tensorboard_python, MOTION,
                 OBJECT, CHECKPOINT]
        for path in files:
            if not path.is_file():
                raise RuntimeError('Required file is missing: %s' % path)
        subprocess.run([str(self.tensorboard_python), '-I', '-B', '-c',
                        'import importlib.util; assert importlib.util.find_spec("tensorboard"), "TensorBoard is not installed"'],
                       check=True, stdin=subprocess.DEVNULL)

    def tensorboard_command(self, port):
        return [str(self.tensorboard_python), '-I', '-B', '-m', 'tensorboard.main',
                '--logdir', str(self.runs), '--host', '127.0.0.1', '--port', str(port)]

    def spawn(self, command, log, tokens):
        for directory in [self.root / 'tmp', self.root / 'cache']:
            directory.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                           WANDB_DISABLED='true', WANDB_MODE='disabled', TMPDIR=str(self.root / 'tmp'),
                           XDG_CACHE_HOME=str(self.root / 'cache'),RESMIMIC_ROOT=str(REPO),
                           RESMIMIC_PYTHON=str(self.tensorboard_python))
        if 'tensorboard.main' in tokens:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        with log.open('ab', buffering=0) as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, cwd=str(self.root), env=environment,
                                       start_new_session=True, close_fds=True)
        self.handles.append(process)
        record = process_identity(process.pid)
        if not record:
            raise RuntimeError('Background child exited before identity capture; inspect %s' % log)
        record.pop('argv')
        record.update(expected_tokens=tokens, log=str(log))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and process.poll() is None:
            if owns_process(record):
                return record
            time.sleep(.01)
        if not owns_process(record):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
            raise RuntimeError('Background child did not retain expected command identity; inspect %s' % log)
        return record

    def wait_tensorboard(self, record, port):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if not owns_process(record):
                raise RuntimeError('TensorBoard exited; inspect %s' % record['log'])
            if owns_listening_port(record['pid'], port):
                return
            time.sleep(.1)
        raise RuntimeError('TensorBoard did not own/listen on port %d; inspect %s' % (port, record['log']))

    def stop_record(self, record, training=False):
        if not owns_process(record):
            return False
        pid = record['pid']
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            return False
        if group != pid:
            raise RuntimeError('Refusing to stop a process outside its recorded private session')
        # Trainer catches SIGTERM, finishes the PPO iteration and saves an
        # interrupted checkpoint. Allow that cooperative path before SIGKILL.
        stages = [(signal.SIGTERM, 30.), (signal.SIGKILL, 2.)] if training else [(signal.SIGTERM, 5.), (signal.SIGKILL, 2.)]
        for sig, timeout in stages:
            if not owns_process(record):
                break
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and owns_process(record):
                time.sleep(.1)
        if owns_process(record):
            raise RuntimeError('Own process still alive after stop attempts: %d' % pid)
        for process in self.handles:
            if process.pid == pid:
                process.wait(timeout=2)
        return True

    def status(self, state):
        training = state.get('training', {})
        tb = state.get('tensorboard', {})
        return dict(training='running' if owns_process(training) else 'not running',
                    training_pid=training.get('pid'), run_dir=state.get('last_run'),
                    training_log=training.get('log'),
                    tensorboard='running' if owns_process(tb) else 'not running',
                    tensorboard_pid=tb.get('pid'), tensorboard_url=tb.get('url'), tensorboard_log=tb.get('log'))

    def start(self, args, state):
        if owns_process(state.get('training')):
            raise RuntimeError('Training is already running (PID %s). Use status or stop first.' % state['training']['pid'])
        self.check_dependencies()
        tb = state.get('tensorboard'); new_tb = False; train = None
        if owns_process(tb):
            if tb.get('port') != args.port or not owns_listening_port(tb['pid'], args.port):
                raise RuntimeError('Own TensorBoard uses another port or is not ready. Use its recorded port, or stop-all first.')
        else:
            # Reject an unrelated listener. Never kill a process to obtain a port.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                try:
                    reservation.bind(('127.0.0.1', args.port))
                except OSError as exc:
                    raise RuntimeError('Port %d is occupied; choose --port with a free port. Existing processes were not changed.' % args.port) from exc
            tb = None
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        run = self.runs / stamp
        run.mkdir(parents=True, exist_ok=False)
        self.runs.mkdir(parents=True, exist_ok=True)
        try:
            if tb is None:
                tb = self.spawn(self.tensorboard_command(args.port), self.state_dir / 'tensorboard.log',
                                ['tensorboard.main', str(self.runs)])
                tb.update(port=args.port, url='http://127.0.0.1:%d' % args.port)
                new_tb = True
                self.wait_tensorboard(tb, args.port)
            command = ['bash', str(self.wrapper), str(self.train_script), 'train', '--num-envs', str(args.num_envs),
                       '--iterations', str(args.iterations), '--seed', str(args.seed), '--output', str(run)]
            if args.num_mini_batches:
                command.extend(['--num-mini-batches', str(args.num_mini_batches)])
            train = self.spawn(command, run / 'training.log', [str(self.train_script), str(run)])
            time.sleep(.3)
            if not owns_process(train):
                raise RuntimeError('Training exited during startup; inspect %s' % (run / 'training.log'))
            state.update(training=train, tensorboard=tb, last_run=str(run), last_start_utc=stamp)
            self.write(state)
            (self.state_dir / 'latest_run.txt').write_text(str(run) + '\n')
            latest_log = self.state_dir / 'training_latest.log'
            temporary_log = self.state_dir / 'training_latest.tmp'
            if temporary_log.is_symlink():
                temporary_log.unlink()
            temporary_log.symlink_to(run / 'training.log')
            temporary_log.replace(latest_log)
        except BaseException:
            if train:
                self.stop_record(train, training=True)
            if new_tb:
                self.stop_record(tb)
            raise
        return self.status(state)

    def execute(self, args):
        if args.command == 'status' and not self.state_dir.exists():
            return self.status({})
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.state_dir / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.read()
            if args.command == 'start':
                return self.start(args, state)
            if args.command in ['stop', 'stop-all']:
                self.stop_record(state.get('training'), training=True)
                if args.command == 'stop-all':
                    self.stop_record(state.get('tensorboard'))
                self.write(state)
            return self.status(state)


def main():
    args = parse_args()
    try:
        result = Controller().execute(args)
        print(json.dumps(result, indent=2))
        if args.command == 'start':
            print('Detached training started. It survives terminal disconnects.')
            print('TensorBoard: %s' % result['tensorboard_url'])
            print('Log: %s' % result['training_log'])
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
