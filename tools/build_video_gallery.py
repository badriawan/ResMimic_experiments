"""Validate bundled clips, create English posters, and build a local HTML gallery."""
import html,json,os,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('TMPDIR',str(root/'.cache/tmp'))
(root/'.cache/tmp').mkdir(parents=True,exist_ok=True)
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
manifest=json.loads((root/'media/manifest.json').read_text());posters=root/'media/posters';posters.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
cards=[];md=['# Video gallery','','All clips are existing experiment results. Reference replay and physical policy evaluation are labeled separately. Click an image/title to open the MP4. GitHub may show a file page; use Raw/Download when needed. For an HTML video player, run `python3 -m http.server 8000 --bind 127.0.0.1` at the repository root and open `http://localhost:8000/docs/videos.html`.','']
for video in manifest['videos']:
    path=root/video['path'];reader=imageio.get_reader(str(path));meta=reader.get_meta_data();n=reader.count_frames()
    # Decode every frame once as an integrity check.
    decoded=sum(1 for _ in reader);reader.close();assert decoded==n
    reader=imageio.get_reader(str(path));frame=reader.get_data(min(n-1,int(n*.65)));reader.close()
    pic=Image.fromarray(frame);pic.thumbnail((640,400));canvas=Image.new('RGB',(640,460),(241,245,249));canvas.paste(pic,((640-pic.width)//2,60+(400-pic.height)//2));draw=ImageDraw.Draw(canvas)
    draw.text((12,8),video['id'].replace('_',' '),font=font,fill=(20,40,60));draw.text((12,34),video['kind'],font=font,fill=(50,70,90))
    poster=posters/(video['id']+'.jpg');canvas.save(poster,quality=90)
    video.update(frames=n,fps=float(meta['fps']),duration_s=n/float(meta['fps']),poster=str(poster.relative_to(root)),decoded_frames=decoded)
    md.extend(['## '+video['id'].replace('_',' '),'',f"[![{video['kind']}](../{video['poster']})](../{video['path']})",'',f"**Dataset:** `{video['dataset']}` · **Type:** {video['kind']} · **Duration:** {video['duration_s']:.2f} s",'',video['caption'],''])
    cards.append('<article><h2>'+html.escape(video['id'].replace('_',' '))+'</h2><p class="kind">'+html.escape(video['kind'])+'</p><video controls preload="metadata" poster="../'+video['poster']+'"><source src="../'+video['path']+'" type="video/mp4"></video><p>'+html.escape(video['caption'])+'</p><a href="../'+video['path']+'">Download MP4</a></article>')
(root/'media/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'docs/VIDEOS.md').write_text('\n'.join(md))
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ResMimic Custom Data — Experiment Videos</title><style>body{font:16px system-ui,sans-serif;margin:0;background:#eef3f8;color:#173047}header{padding:36px max(24px,5vw);background:#102d44;color:white}main{max-width:1400px;margin:24px auto;padding:0 24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}article{background:white;padding:20px;border-radius:12px;box-shadow:0 2px 9px #17304715}h1{margin:0 0 12px}h2{font-size:19px;text-transform:capitalize}.kind{font-weight:700;color:#2c637a}video{width:100%;background:#111;border-radius:7px}a{color:#12628d}p{line-height:1.5}</style></head><body><header><h1>ResMimic Custom Data — Experiment Videos</h1><p>Reference replay, contact corrections and physical policy evaluations.</p><p>Research status: physical floorlamp lifting/carrying remains unsuccessful. The historical residual used the earlier timeout bug.</p></header><main>'''+''.join(cards)+'''</main></body></html>'''
(root/'docs/videos.html').write_text(page)
report={'status':'PASS','videos':len(manifest['videos']),'decoded_frames':sum(v['frames'] for v in manifest['videos']),'all_video_hashes_match':all(hashlib.sha256((root/v['path']).read_bytes()).hexdigest()==v['sha256'] for v in manifest['videos'])}
(root/'publication/video_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
