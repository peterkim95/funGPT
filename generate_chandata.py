import json
import html
import re

from tqdm import tqdm

CLENR = re.compile('<.*?>')

def clean_html(raw_html):
    return re.sub(CLENR, '', raw_html)

def get_original_post_no(text):
    return int(text[2:10]) if text.startswith('>>') else None

def filter_leading(text):
    return text[10:] if text.startswith('>>') else text

result = []
with open('pol_062016-112019_labeled.ndjson', 'r') as f:
    for line in tqdm(f, total=3397911): # wc -l'ed beforehand
        t = json.loads(line)
        msg = {}
        for p in t['posts']:
            # print(p)
            if p['resto'] == 0 and 'com' not in p:
                break
            if 'com' in p:
                no = p['no']
                text = html.unescape(p['com'])
                text = text.replace('<br>', ' ')
                text = clean_html(text)
                msg[no] = text

                # print(no, text)

                if p['resto'] == 0: # first post doesn't have a source post
                    continue

                post = filter_leading(msg[p['resto']])
                reply = filter_leading(text)
                
                try:
                    opost_no = get_original_post_no(text)
                except:
                    break
                if opost_no:
                    if opost_no not in msg: # this reply doesn't refer to any post in the current posts
                        continue
                    post = filter_leading(msg[opost_no])
                
                result.append({'post': post, 'reply': reply})
        # print('-'*100)
        # to not explode ram
        if len(result) >= 5200000:
            break

with open('chan_data.json', 'w+') as f:
    json.dump(result, f, indent=4, default=str)