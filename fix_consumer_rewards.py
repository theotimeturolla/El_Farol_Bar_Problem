# Modifier environment/multi_bars_env.py pour forcer la sortie

import re

with open('environment/multi_bars_env.py', 'r') as f:
    content = f.read()

# Trouver la section rewards consumers
old_pattern = r"if choice == 0:  # Stayed home\s+rewards\[consumer_agent\] = 5\.0"

# Nouveau reward : PÉNALITÉ pour rester à la maison
new_code = """if choice == 0:  # Stayed home
                rewards[consumer_agent] = -5.0  # PENALTY for staying home"""

content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)

with open('environment/multi_bars_env.py', 'w') as f:
    f.write(content)

print(" Staying home now gives PENALTY (-5.0 instead of +5.0)")
print("   Consumers will be FORCED to go out!")
