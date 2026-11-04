from __future__ import annotations

import argparse, csv, hashlib, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent.parent
OUT_IMAGES = ROOT / 'dataset' / 'week6_style_4500_images'
OUT_CSV = ROOT / 'dataset' / 'week6_style_4500_labels.csv'

STYLE_LABELS = ['Elegant','Casual','Romantic','Minimalist','Sporty','Streetwear','Boho','Office','Evening']

EXACT_DS = 'lihicarmeli/fashion-stylist-multimodal-v2'
BIAS_DS = 'shaghayegh/stylistic-bias-dataset'

EXACT_MAP = {
    'elegant':'Elegant','casual':'Casual','romantic':'Romantic','minimalist':'Minimalist',
    'sporty':'Sporty','streetwear':'Streetwear','boho':'Boho','office':'Office'
}
BIAS_MAP = {
    'Casual':'Casual',
    'Sporty / Athletic wear':'Sporty',
    'Streetwear':'Streetwear',
    'Professional / Business formal':'Office',
    'Formal / Evening wear':'Evening',
}

def split_exact(key: str) -> str:
    n = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100
    return 'train' if n < 80 else ('validation' if n < 90 else 'test')

def add(store, style, image, source, source_id, split):
    if style and image is not None:
        store[style].append({
            'image': image.convert('RGB'), 'source': source,
            'source_id': str(source_id), 'split': split,
        })

def load_sources(store):
    print('1/2 Loading exact project styles:', EXACT_DS)
    ds = load_dataset(EXACT_DS, split='train')
    for i, row in enumerate(ds):
        style = EXACT_MAP.get(str(row.get('style_preference','')).strip().lower())
        if not style:
            continue
        quality = str(row.get('image_quality','good')).strip().lower()
        if quality not in ('', 'good'):
            continue
        image = row.get('image_improved') or row.get('image')
        rid = str(row.get('id', i))
        add(store, style, image, EXACT_DS, rid, split_exact('exact:'+rid))

    print('2/2 Loading full-body fashion styles:', BIAS_DS)
    for hf_split, split in [('train','train'),('validation','validation'),('test','test')]:
        ds = load_dataset(BIAS_DS, split=hf_split, streaming=True)
        for row in ds:
            if str(row.get('variation_attribute','')).strip() != 'fashion_style':
                continue
            if str(row.get('asset_kind','')).strip() != 'body':
                continue
            style = BIAS_MAP.get(str(row.get('variation_value','')).strip())
            if not style:
                continue
            sid = f"face{row.get('face_id','')}_var{row.get('variation_id','')}"
            add(store, style, row.get('image'), BIAS_DS, sid, split)

def augment(img: Image.Image, k: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    out = img.copy()
    if k % 2 == 0:
        out = ImageOps.mirror(out)
    w, h = out.size
    p = 0.02 + 0.01 * (k % 4)
    dx, dy = max(1,int(w*p)), max(1,int(h*p))
    out = out.crop((dx,dy,w-dx,h-dy)).resize((w,h), Image.Resampling.LANCZOS)
    out = ImageEnhance.Brightness(out).enhance(0.95 + rng.random()*0.10)
    out = ImageEnhance.Contrast(out).enhance(0.95 + rng.random()*0.10)
    return out

def choose(items, target, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for x in items:
        by[x['split']].append(x)
    for v in by.values():
        rng.shuffle(v)

    total = sum(map(len, by.values()))
    if total <= target:
        return by['train'] + by['validation'] + by['test']

    chosen = by['validation'][:50] + by['test'][:50]
    need = target - len(chosen)
    chosen += by['train'][:need]
    need = target - len(chosen)
    if need > 0:
        chosen += (by['validation'][50:] + by['test'][50:])[:need]
    return chosen[:target]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-style', type=int, default=500)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    if OUT_IMAGES.exists():
        shutil.rmtree(OUT_IMAGES)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    store = {s: [] for s in STYLE_LABELS}
    load_sources(store)

    rows = []
    original_counts = {}
    aug_counts = Counter()

    for si, style in enumerate(STYLE_LABELS):
        originals = choose(store[style], args.per_style, args.seed + si)
        original_counts[style] = len(originals)
        train_base = [x for x in originals if x['split'] == 'train']
        if not train_base:
            raise RuntimeError(f'No training originals for {style}')

        slug = style.lower().replace(' ','_')
        n = 0
        for x in originals:
            n += 1
            fn = f'{slug}_{n:04d}.jpg'
            x['image'].save(OUT_IMAGES/fn, 'JPEG', quality=94)
            rows.append({
                'file':fn,'style':style,'split':x['split'],'source':x['source'],
                'source_id':x['source_id'],'is_augmented':0,'augmentation_of':''
            })

        missing = args.per_style - len(originals)
        for k in range(missing):
            base = train_base[k % len(train_base)]
            img = augment(base['image'], k, args.seed + si*100000 + k)
            n += 1
            fn = f'{slug}_{n:04d}.jpg'
            img.save(OUT_IMAGES/fn, 'JPEG', quality=94)
            rows.append({
                'file':fn,'style':style,'split':'train','source':base['source'],
                'source_id':f"aug:{base['source_id']}:{k}",'is_augmented':1,
                'augmentation_of':base['source_id']
            })
            aug_counts[style] += 1

    random.Random(args.seed).shuffle(rows)
    with OUT_CSV.open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file','style','split','source','source_id','is_augmented','augmentation_of'])
        w.writeheader(); w.writerows(rows)

    counts = Counter(r['style'] for r in rows)
    splits = Counter((r['style'],r['split']) for r in rows)
    print('\nDONE')
    print('Labels:', OUT_CSV)
    print('Images:', OUT_IMAGES)
    print('Total samples:', len(rows))
    for s in STYLE_LABELS:
        print(f"{s}: {counts[s]} | originals={original_counts[s]} | augmented={aug_counts[s]} | train={splits[(s,'train')]} val={splits[(s,'validation')]} test={splits[(s,'test')]}")
    print('\nValidation/test contain ORIGINAL images only.')

if __name__ == '__main__':
    main()
