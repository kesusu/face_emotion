import json, os
p = os.path.join(os.path.dirname(__file__), 'RAF-DB', 'train', 'reports', 'rf_cleaned_v2_report.json')
if os.path.exists(p):
    d = json.load(open(p, 'r', encoding='utf-8'))
    print(json.dumps(d, indent=2, ensure_ascii=False))
else:
    print('REPORT_NOT_FOUND at:', p)
    # list what IS there
    parent = os.path.dirname(p)
    if os.path.exists(parent):
        files = [f for f in os.listdir(parent) if f.endswith('.json')]
        print('Available json files:', files)
