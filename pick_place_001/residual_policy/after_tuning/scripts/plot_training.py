"""Export the most recent run's actual TensorBoard scalar curves."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,install,write_json
install()
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
run=Path((ROOT/'state/latest_run.txt').read_text().strip())
acc=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});acc.Reload()
tags=acc.Tags()['scalars']
wanted=['Train/mean_reward','Train/mean_episode_length','Loss/value_func','Loss/surrogate','Object/position_error_m','Object/point_error_m']
curves={tag:[dict(iteration=e.step+1,value=e.value) for e in acc.Scalars(tag)] for tag in wanted if tag in tags}
write_json(ROOT/'reports/latest_training_curves.json',dict(run=str(run),available_tags=tags,curves=curves))
fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
for ax,tag in zip(axes.flat,wanted):
    rows=curves.get(tag,[])
    if rows:ax.plot([e['iteration'] for e in rows],[e['value'] for e in rows],color='#2375a6',linewidth=1)
    ax.set_title(tag);ax.set_xlabel('Completed PPO iteration');ax.grid(alpha=.25)
fig.suptitle('Latest residual training: recorded metrics, physical result evaluated separately')
fig.savefig(ROOT/'reports/latest_training_curves.png',dpi=150);plt.close(fig)
print(json.dumps({tag:dict(samples=len(rows),first=rows[0],last=rows[-1]) for tag,rows in curves.items()},indent=2))
