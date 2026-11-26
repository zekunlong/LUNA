"""Download author-hosted EuroSAT RGB and the unchanged CoOp-linked split."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile

import requests
from PIL import Image

ROOT = Path('E:/LUNA/data')
OUT = Path('E:/LUNA/results_reliability_analysis/logs/asset_acquisition_eurosat.json')
RGB_URL = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
SPLIT_URL = 'https://drive.usercontent.google.com/download?id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o&export=download&confirm=t'
MD5 = 'f46e308c4d50d4bf32fedad2d3d62f3b'

def hashes(path):
    sha, md = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            sha.update(chunk); md.update(chunk)
    return {'size_bytes': path.stat().st_size, 'sha256': sha.hexdigest(), 'md5': md.hexdigest()}

def download(url, path):
    if path.exists():
        print('Reusing existing file for verification: '+str(path), flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial=path.with_name(path.name+'.partial_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    started=time.monotonic(); reported=started; count=0
    with requests.get(url, stream=True, timeout=(30,120)) as response:
        response.raise_for_status()
        with partial.open('xb') as stream:
            for chunk in response.iter_content(1024*1024):
                if chunk:
                    stream.write(chunk); count+=len(chunk)
                if time.monotonic()-reported>15:
                    print(f'Download {path.name}: {count:,} bytes', flush=True); reported=time.monotonic()
    partial.rename(path)
    print(f'Download complete {path.name}: {count:,} bytes', flush=True)
    return path

def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    archive=ROOT/'downloads/EuroSAT_RGB.zip'
    split=ROOT/'eurosat/split_zhou_EuroSAT.json'
    receipt={'status':'DOWNLOADING','dataset':'EuroSAT','started_utc':datetime.now(timezone.utc).isoformat(),
             'data_root':str(ROOT),'dataset_root':str(ROOT/'eurosat'),'image_root':str(ROOT/'eurosat/2750'),'split_path':str(split)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(download,RGB_URL,archive),pool.submit(download,SPLIT_URL,split)]
        for future in futures: future.result()
    archive_hash=hashes(archive)
    if archive_hash['md5']!=MD5: raise ValueError('Official RGB archive MD5 mismatch; archive retained for diagnosis')
    records=json.loads(split.read_text(encoding='utf-8-sig'))
    if set(records)!=set(('train','val','test')): raise ValueError('Original split schema invalid')
    image_root=ROOT/'eurosat/2750'
    image_root.mkdir(parents=True,exist_ok=True)
    roots=set(); extracted=0; preserved=0
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            if item.is_dir() or not item.filename.lower().endswith(('.jpg','.jpeg')): continue
            parts=PurePosixPath(item.filename).parts
            if '..' in parts or item.filename.startswith('/'): raise ValueError('Unsafe archive path')
            roots.add(parts[0])
            relative=Path(*parts[-2:])
            target=(image_root/relative).resolve()
            if not target.is_relative_to(image_root.resolve()): raise ValueError('Archive path outside target')
            data=package.read(item)
            if target.exists():
                if target.read_bytes()!=data: raise ValueError('Existing image differs; refusing overwrite: '+str(target))
                preserved+=1
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream: stream.write(data)
                extracted+=1
            if (extracted+preserved)%5000==0: print(f'Extracted/verified {extracted+preserved} images',flush=True)
    paths=list(image_root.glob('*/*.jpg'))
    modes={}; sizes={}; class_counts={}
    for index,path in enumerate(paths,1):
        with Image.open(path) as picture:
            picture.load()
            modes[picture.mode]=modes.get(picture.mode,0)+1
            key=str(tuple(picture.size)); sizes[key]=sizes.get(key,0)+1
        class_counts[path.parent.name]=class_counts.get(path.parent.name,0)+1
        if index%5000==0: print(f'Decoded/verified {index} RGB images',flush=True)
    if len(paths)!=27000 or len(class_counts)!=10 or set(modes)!={'RGB'}: raise ValueError('Unexpected EuroSAT RGB contents')
    seen={}; split_sets={}; missing=[]; duplicates=0; labels={}
    for name,rows in records.items():
        split_sets[name]=set()
        for reference,label,classname in rows:
            path=(image_root/reference).resolve()
            if not path.is_relative_to(image_root.resolve()): raise ValueError('Unsafe split image reference')
            if not path.is_file(): missing.append(str(path))
            key=str(path)
            if key in seen: duplicates+=1
            seen[key]=name; split_sets[name].add(key)
            if label in labels and labels[label]!=classname: raise ValueError('Conflicting class names')
            labels[label]=classname
    overlaps={f'{a}/{b}':len(split_sets[a]&split_sets[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
    if missing or duplicates or len(seen)!=27000: raise ValueError('Split missing/duplicate/incomplete images')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from experiments.journal_preflight import _inspect_dataset
    blockers=[]; checked=_inspect_dataset('eurosat',ROOT,blockers)
    if blockers: raise ValueError('; '.join(blockers))
    receipt.update(status='READY_DATASET_AND_ORIGINAL_SPLIT',completed_utc=datetime.now(timezone.utc).isoformat(),
                   archive={'path':str(archive),'url':RGB_URL,'official_link_source':'https://github.com/phelber/EuroSAT',
                            'checksum_source':'https://zenodo.org/records/7711810','published_md5':MD5,'checksum_verified':True,
                            'archive_top_level_names':sorted(roots),**archive_hash},
                   split={'path':str(split),'url':SPLIT_URL,'official_link_source':'https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md',
                          'google_drive_file_id':'1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o','generated':False,**hashes(split)},
                   verification={'image_count':len(paths),'class_count':len(class_counts),'class_image_counts':class_counts,
                                 'image_modes':modes,'image_sizes':sizes,'all_images_decodable':True,
                                 'split_counts':{key:len(value) for key,value in records.items()},
                                 'missing_image_count':0,'duplicate_reference_count':0,'cross_split_overlap_counts':overlaps,
                                 'split_classnames':labels,'new_images_extracted':extracted,'existing_images_preserved':preserved},
                   dataset_preflight=checked,dataset_blockers=[])
    OUT.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        receipt=json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
        receipt.update(status='FAILED_ASSET_ACQUISITION',error=type(exc).__name__+': '+str(exc))
        OUT.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        raise
