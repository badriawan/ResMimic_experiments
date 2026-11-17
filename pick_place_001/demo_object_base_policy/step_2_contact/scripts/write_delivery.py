"""Package a geometry-validated reference with explicit static/dynamic status."""
import argparse,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,PICK,MOTION,CHECKPOINT,install,sha,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args();install();candidate=args.candidate.resolve();tag=candidate.parent.name
    if ROOT not in candidate.parents:raise ValueError('Candidate outside step_2')
    geometry=json.loads((ROOT/('reports/independent_'+tag+'_validation.json')).read_text());loader=json.loads((ROOT/('reports/loader_'+tag+'.json')).read_text());wrench=json.loads((ROOT/('reports/wrench_'+tag+'.json')).read_text())
    assert all(geometry['checks'].values()) and loader['status']=='PASS','Geometry/loader must pass before packaging'
    static=wrench['summaries']['projected_patch_static'];dynamic=wrench['summaries']['projected_patch_dynamic'];static_ok=static['gravity_plus_contact_bounded_equilibrium_frames']==static['frames']
    assert static_ok,'A geometry-only reference must not be promoted as statically supportive'
    baseline=json.loads((PICK/'baseline_manifest.json').read_text())
    assert sha(MOTION)==baseline['robot_motion']['sha256'] and sha(MOTION.with_suffix('.npz'))==baseline['object_motion']['sha256'] and sha(CHECKPOINT)==baseline['base_policy']['sha256']
    out=ROOT/'reference_corrected';out.mkdir(exist_ok=False)
    for path in [candidate,candidate.with_suffix('.npz'),candidate.parent/'contact_plan.json',candidate.parent/'solver_report.json',candidate.parent/'solver_targets.npz']:shutil.copyfile(path,out/path.name)
    selection=dict(status='GEOMETRY_PASS_CONDITIONAL_STATIC_SUPPORT_PASS',candidate=str(candidate),packaged_robot=str(out/candidate.name),packaged_object=str(out/candidate.with_suffix('.npz').name),geometry_report='reports/independent_'+tag+'_validation.json',loader_report='reports/loader_'+tag+'.json',wrench_report='reports/wrench_'+tag+'.json',static_support='CONDITIONAL_ON_MAINTAINED_PROJECTED_PATCH_CONTACTS',dynamic_wrench_model='PASS' if dynamic['airborne_gravity_plus_contact_bounded_frames']==dynamic['reference_airborne_frames'] else 'FAIL_AT_ORIGINAL_SPEED',dynamic_carrying_verified=False,policy_executed=False,training_executed=False,active_baseline_replaced=False,files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write_json(ROOT/'selected_reference.json',selection)
    g=geometry['metrics'];fmt=lambda a:', '.join('%.3f'%x for x in a)
    object_gap=g['pad_surface_gap_range_m'];conditioning=loader.get('conditioning_diagnostics',{})
    text=f'''# Step 2 — koreksi reference tangan dan floorlamp

**Geometri reference lulus dan dukungan statis bersyarat lulus. Status model gaya dinamis: {selection['dynamic_wrench_model']}.** Hasil ini belum merupakan keberhasilan membawa objek dengan base policy atau simulator fisika.

Pasangan data yang dikoreksi ada di [reference_corrected/{candidate.name}](reference_corrected/{candidate.name}) dan [reference_corrected/{candidate.with_suffix('.npz').name}](reference_corrected/{candidate.with_suffix('.npz').name}). Keduanya tersinkron, {geometry['frames']} frame pada {geometry['fps']:g} Hz, endpoint 4,1667 detik. File sumber P1 attempt09 dan checkpoint tetap utuh; baseline aktif tidak diganti.

## Apa yang diperbaiki

G1 memiliki 29 joint aktif dan tangan karet tetap; kedua joint palm bersifat fixed, tanpa jari/gripper aktif. Reference baru menggunakan bidang samping convex hull tangan yang sebenarnya, menekan dua facet berlawanan dari collider batang lampu. Material anchor dan normalnya disimpan dalam [contact_plan.json](reference_corrected/contact_plan.json). Bidang ini jauh lebih luas daripada facet kecil bagian atas tangan pada percobaan awal.

Robot dan objek dirancang sebagai satu kesatuan: XY/yaw lampu dioptimalkan bersama lengan untuk menemukan konfigurasi yang dapat dijangkau. Lintasan objek kemudian dihaluskan dengan lowpass 100 ms dan spline C2, dibekukan, lalu 14 joint lengan diproyeksikan kembali secara nonlinear. Root, 12 joint kaki, dan waist mempertahankan lintasan posisi P1; semua knot asli tetap identik. Seluruh FK dihitung ulang.

Ekspor diperhalus dari 120 ke 480 Hz untuk menghindari penetrasi akibat interpolasi joint antara frame. Durasi tidak diperpanjang. Data NPZ menyimpan `source_time_s` yang tetap memetakan waktu sumber. Peningkatan FPS mengubah hasil turunan velocity dan target skeleton yang dihitung loader; ini bukan jaminan bahwa semua observasi policy identik dengan dataset lama. Angka perbedaannya ada di [laporan loader](reports/loader_{tag}.json).

Strategi dimulai **pre-grasped sejak frame pertama**. Fase mendekati atau mengambil lampu dari lantai tidak disertifikasi. Tidak ada constraint penopang atau perubahan aset robot/lampu untuk memalsukan pegangan.

## Validasi independen

| Pemeriksaan | Hasil |
|---|---|
| Geometri | {len(geometry['checks'])} pemeriksaan PASS, {geometry['dense_samples']} sampel pada {geometry['dense_rate_hz']:g} Hz |
| Error posisi pad maksimum kiri/kanan | {fmt([1000*x for x in g['max_pad_error_m']])} mm |
| Error normal maksimum kiri/kanan | {fmt(g['max_normal_error_deg'])}° |
| Gap pad normal minimum kiri/kanan | {fmt([1000*x for x in object_gap[0]])} mm |
| Gap pad normal maksimum kiri/kanan | {fmt([1000*x for x in object_gap[1]])} mm |
| Self-clearance minimum | {1000*g['minimum_self_clearance_m']:.3f} mm |
| Clearance lampu–bagian robot selain tangan | {1000*g['minimum_lamp_nonhand_clearance_m']:.3f} mm |
| Drift tangensial pad kiri/kanan | {fmt(g['maximum_material_tangent_drift_deg'])}° |
| Rasio kecepatan joint terhadap limit | {g['maximum_joint_velocity_ratio']:.4f} |
| Posisi root/kaki/waist dan knot P1 | PASS — termasuk bukti lintasan kaki kontinu |
| Loader ResMimic HOI + adapter lokal | PASS — {loader['samples']} sampel, CPU; tanpa simulator/policy |

Model collision untuk audit memakai hull/primitif sumber, tiga collider lampu yang identik dengan step 0, serta pendekatan capsule untuk cylinder robot. Ini bukan bukti kesetaraan persis dengan hasil cooking PhysX. Query CCD nol yang tidak konsisten tidak diterima sebagai kontak; oracle memakai certificate support-plane/QP yang valid. Tidak ada gate penetrasi yang dilonggarkan.

Audit reference lama menemukan penetrasi source-hull tangan–lampu sampai sekitar 66,8 mm kiri dan 77,8 mm kanan, serta tangan–tangan 38,6 mm. Perbandingan ini bersifat geometri, bukan pengukuran gaya simulasi.

## Dukungan statis dan batas dinamika

Perhitungan memakai massa **4 kg**, friction **0,6**, COM/inertia dari aset step 0, dan limit torque efektif robot yang direkam pada step 1. Massa/material tetap asumsi, bukan pengukuran benda nyata.

LP memakai cone friction delapan arah dan patch berupa irisan proyeksi segitiga hull tangan dengan facet batang yang terbatas. Patch hanya dapat digunakan jika kontak tersebut benar-benar terbentuk setelah gap geometris ditutup; tekanan aktual tidak disimulasikan.

- Dukungan statis dengan batas torque dan gravitasi URDF: **{static['gravity_plus_contact_bounded_equilibrium_frames']}/{static['frames']} pose feasible secara bersyarat**.
- Dukungan dinamis pada sampel airborne dengan model yang sama: **{dynamic['airborne_gravity_plus_contact_bounded_frames']}/{dynamic['reference_airborne_frames']} pose feasible**.
- Normal gaya dan residual per frame tersedia di [wrench_{tag}.json](reports/wrench_{tag}.json).

Hasil LP bukan sertifikasi inverse dynamics seluruh tubuh: tidak memasukkan inersia robot, reaksi tanah, keseimbangan berjalan, kontak yang benar-benar dihasilkan controller, maupun dinamika tekanan/friksi saat patch berputar. Model dua hard-point berlawanan memiliki rank lima dan tidak dapat dianggap full grasp closure. Rasio torque dari solusi minimum-normal-force hanyalah satu alokasi; keputusan feasibility memakai LP terpisah yang dibatasi torque, termasuk gravity bias.

**Jika model dinamis gagal, gunakan hasil ini sebagai reference geometri dan kontak statis bersyarat, bukan reference yang sudah dinyatakan siap membawa 4 kg pada kecepatan asli.** Perlu perencanaan gaya/gerakan dan uji objek bebas berikutnya. Perbaikan reference ini juga tidak menyelesaikan error locomotion base policy yang ditemukan pada step 1.

## Hasil dan skrip

- [Video reference seluruh tubuh dan detail tangan](results/reference_{tag}/reference_corrected.mp4)
- [Grafik kontak sebelum/sesudah](reports/contact_diagnostics.png)
- [Validasi geometri](reports/independent_{tag}_validation.json)
- [Ringkasan pemilihan data](selected_reference.json)
- [Model spline objek dan penjelasan smoothing](reports/solver_temporal_attempt12.json)
- [State target kontak](reference_corrected/solver_targets.npz)
- [Laporan solver](reference_corrected/solver_report.json)
- [Review akhir](reports/final_review.json)
- [Manifest file dan hash](reports/package_manifest.json)

Video dibuat melalui **renderer MuJoCo sebagai replay pose reference**; tidak menjalankan base policy, training, atau dinamika lift. Collider ditampilkan transparan hijau karena diameter collider batang berbeda dari mesh visual. Marker menunjukkan target kontak, bukan sensor gaya.

Dari root repository, contoh pemeriksaan ulang:

```bash
thirdparty/gmr_omomo/bin/python -I -B \\
  InterPose/custom_data/pick_place_001/demo_object_base_policy/step_2/scripts/validate_reference.py \\
  --candidate "$PWD/InterPose/custom_data/pick_place_001/demo_object_base_policy/step_2/reference_corrected/{candidate.name}" \\
  --rate 2400 --tag delivered_check

InterPose/custom_data/pick_place_001/demo_object_base_policy/step_2/check_loader.sh \\
  --candidate "$PWD/InterPose/custom_data/pick_place_001/demo_object_base_policy/step_2/reference_corrected/{candidate.name}"
```

Perintah generator dan seluruh argumen run tersimpan di laporan solver/temporal serta source skrip. `solve_contact_reference.py` menangani pencarian coupled; `solver_temporal.py`/helper dense melakukan smoothing dan reprojection (lihat nama skrip aktual di manifest). Semua log/kandidat awal dipertahankan untuk audit. Attempt09 dan attempt11 tidak dipromosikan karena gagal di antara frame; attempt10 hanya fallback geometri dengan dukungan gaya yang tidak lolos. Loader attempt08 awal gagal karena toleransi float32 terlalu ketat; hasil itu disimpan, dan pemeriksaan final memakai bound presisi waktu/phase float32 yang eksplisit.

Semua file baru, log, cache, aset turunan, dan hasil berada di `step_2`. Aset dasar robot dan data sumber tetap dibaca dari repository; paket ini membutuhkan checkout project yang sama untuk dijalankan ulang.
'''
    (ROOT/'README.md').write_text(text)
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
