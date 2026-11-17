"""Compare the same frozen-object reference before/after arm smoothing at60FPS."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,install,sha,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--after-video',type=Path,required=True);args=p.parse_args();install();after=args.after_video.resolve()
    if ROOT not in after.parents:raise ValueError('Video outside arm_smoothing')
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    before=ROOT/'results/before/reference_corrected.mp4';sources=[before,after];readers=[imageio.get_reader(str(v)) for v in sources];metadata=[r.get_meta_data() for r in readers]
    assert all(abs(m['fps']-60)<1e-6 and tuple(m['size'])==(1920,720) for m in metadata)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26);output=ROOT/'reports/arm_smoothing_comparison.mp4';writer=imageio.get_writer(str(output),fps=60,codec='libx264',quality=8,macro_block_size=16)
    chosen={0,134,247};stills=[];count=0
    try:
        while True:
            frames=[]
            for r in readers:
                try:frames.append(r.get_data(count))
                except IndexError:frames.append(None)
            if all(f is None for f in frames):break
            assert all(f is not None for f in frames),'Different source durations'
            canvas=Image.new('RGB',(1920,768),'#edf2f5');draw=ImageDraw.Draw(canvas)
            draw.text((18,9),'SEBELUM | reference terakhir',font=font,fill='#182c3e');draw.text((978,9),'SESUDAH | smoothing lengan',font=font,fill='#182c3e')
            for col,frame in enumerate(frames):canvas.paste(Image.fromarray(frame[:,960:1920,:]),(col*960,48))
            writer.append_data(np.asarray(canvas))
            if count in chosen:
                pic=canvas.resize((960,384));stills.append((count,pic))
            count+=1
    finally:
        writer.close()
        for r in readers:r.close()
    r=imageio.get_reader(str(output));decoded=sum(1 for f in r);r.close();assert decoded==count==251
    sheet=Image.new('RGB',(960,404*len(stills)),'#edf2f5');draw=ImageDraw.Draw(sheet)
    for row,(index,pic) in enumerate(stills):sheet.paste(pic,(0,row*404));draw.text((14,row*404+387),'t=%.4fs | sourceframe%d'%(index/60,index*8),fill='#182c3e')
    sheet.save(ROOT/'reports/video_comparison_sheet.jpg',quality=94)
    write_json(ROOT/'reports/video_comparison_validation.json',dict(status='PASS',decoded_frames=decoded,fps=60,dimensions=[768,1920,3],source_video_sha256={str(v):sha(v) for v in sources},sha256=sha(output),alignment='Same times/native480Hz indices0,8,...2000; same camera/object/root. Hand detail crop of both references.',scope='MuJoCo pose replay only; no policy/dynamic carry test'))
    print('PASS comparison video',decoded)

if __name__=='__main__':main()
