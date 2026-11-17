"""Decode matching before/after timestamps for a visual review sheet."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,install
install()
import imageio.v2 as imageio
from PIL import Image,ImageDraw
old=imageio.get_reader(str(ROOT/'videos/lift_hold_001_reference_front.mp4'))
new=imageio.get_reader(str(ROOT/'videos/lift_hold_001_remove_vibration_final.mp4'))
canvas=Image.new('RGB',(1280,6*500),(245,245,245));draw=ImageDraw.Draw(canvas)
for row,t in enumerate([.7,1.1,3.,5.2,5.65,6.]):
 for col,(reader,label) in enumerate([(old,'BEFORE'),(new,'AFTER')]):
  frame=Image.fromarray(reader.get_data(round(t*30))).resize((640,480))
  canvas.paste(frame,(col*640,row*500+20));draw.text((col*640+10,row*500+3),'%s  t=%.1fs'%(label,t),fill=(0,0,0))
canvas.save(ROOT/'videos/remove_vibration_comparison_final.jpg')
with imageio.get_writer(str(ROOT/'videos/lift_hold_001_before_after.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
 for frame in range(382):
  pic=Image.new('RGB',(1280,512),(245,245,245));draw=ImageDraw.Draw(pic)
  for col,(reader,label) in enumerate([(old,'BEFORE'),(new,'AFTER')]):
   pic.paste(Image.fromarray(reader.get_data(frame)).resize((640,480)),(col*640,32));draw.text((col*640+10,8),label,fill=(0,0,0))
  import numpy as np
  writer.append_data(np.asarray(pic))
old.close();new.close()
