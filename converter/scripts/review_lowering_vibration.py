"""Decode and compare the prior hold-only reference and the lowering correction."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,install,sha,write_json
install()
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
paths=[ROOT/'videos/lift_hold_001_remove_vibration_final.mp4',ROOT/'videos/lift_hold_001_lowering_v2_final.mp4']
readers=[imageio.get_reader(str(p)) for p in paths]
marks=[5.7,6.2,7.5,8.5,10.3,10.7,11.3,11.5,12.7]
canvas=Image.new('RGB',(1280,9*500),(245,245,245));draw=ImageDraw.Draw(canvas)
for row,t in enumerate(marks):
 for col,(reader,label) in enumerate(zip(readers,['HOLD ONLY','HOLD + LOWERING'])):
  frame=Image.fromarray(reader.get_data(round(t*30))).resize((640,480))
  canvas.paste(frame,(col*640,row*500+20));draw.text((col*640+10,row*500+3),'%s  t=%.2fs'%(label,t),fill=(0,0,0))
canvas.save(ROOT/'videos/lowering_v2_final_review.jpg')
out=ROOT/'videos/lift_hold_001_lowering_before_after_final.mp4'
if out.exists():raise FileExistsError(out)
with imageio.get_writer(str(out),fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
 for k in range(382):
  pic=Image.new('RGB',(1280,512),(245,245,245));draw=ImageDraw.Draw(pic)
  for col,(reader,label) in enumerate(zip(readers,['HOLD ONLY','HOLD + LOWERING'])):
   pic.paste(Image.fromarray(reader.get_data(k)).resize((640,480)),(col*640,32));draw.text((col*640+10,8),label,fill=(0,0,0))
  writer.append_data(np.asarray(pic))
for reader in readers:reader.close()
check=imageio.get_reader(str(out));count=0
for frame in check:
 assert frame.shape==(512,1280,3);count+=1
check.close();assert count==382
write_json(ROOT/'reports/lowering_comparison_video_final.json',dict(status='PASS_DECODE',path=str(out),sha256=sha(out),frames=count,fps=30,inputs={str(p):sha(p) for p in paths},review_image=str(ROOT/'videos/lowering_v2_final_review.jpg'),review_timestamps_s=marks))
