import re
fnfc_wins = 0
prom_wins = 0
fnfc_total = 0
prom_total = 0
current_ds = ''
with open('final_stage_comparison.md', 'r', encoding='utf-8') as f:
    for line in f:
        if 'Dataset: FNFC' in line:
            current_ds = 'FNFC'
        elif 'Dataset: PROMISE' in line:
            current_ds = 'PROMISE'
        elif '| **' in line:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 5:
                base_str = parts[2].replace('%', '')
                p1a_str = parts[3].replace('%', '')
                if base_str != '-' and p1a_str != '-':
                    try:
                        base = float(base_str)
                        p1a = float(p1a_str)
                        if current_ds == 'FNFC':
                            fnfc_total += 1
                            if p1a > base: fnfc_wins += 1
                        elif current_ds == 'PROMISE':
                            prom_total += 1
                            if p1a > base: prom_wins += 1
                    except:
                        pass
print(f'FNFC: {fnfc_wins} out of {fnfc_total} configurations beat the base paper.')
print(f'PROMISE: {prom_wins} out of {prom_total} configurations beat the base paper.')
