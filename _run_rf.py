import os
os.chdir(r'c:\Users\ke\Desktop\嵌赛\face_emotion')

try:
    exec(open('_rf_cleaned_retrain.py', encoding='utf-8').read())
except Exception as e:
    with open('_rf_error_log.txt', 'w', encoding='utf-8') as f:
        f.write(f"ERROR: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc(file=f)
