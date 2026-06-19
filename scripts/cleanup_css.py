import re

css_path = r"c:\Users\yassi\Downloads\DentAlign (2)\DentAlign\DentAlign\static\css\style.css"

with open(css_path, "r", encoding="utf-8") as f:
    css = f.read()

# 1. Remove the :root block from style.css since we rely on dentalign-theme.css
css = re.sub(r":root\s*\{[^}]+\}", "/* :root removed, see dentalign-theme.css */", css, count=1)

# 2. Replace hardcoded colors with variables
replacements = [
    (r"#2f6fed", "var(--primary)"),
    (r"#1d4ed8", "oklch(0.45 0.16 250)"),
    (r"#0f172a", "var(--foreground)"),
    (r"#64748b", "var(--muted-foreground)"),
    (r"#e6ecff", "var(--border)"),
    (r"#f4f7ff", "var(--background)"),
    (r"#ffffff", "var(--card)"),
]

for old, new in replacements:
    css = re.sub(old, new, css, flags=re.IGNORECASE)

# 3. Clean up duplicate button definitions
# It's tricky to remove all duplicates safely without a CSS parser, 
# but we can replace linear-gradient with var(--gradient-primary)
css = re.sub(
    r"linear-gradient\(180deg,\s*var\(--primary\),\s*oklch\(0\.45 0\.16 250\)\)", 
    "var(--gradient-primary)", 
    css
)

with open(css_path, "w", encoding="utf-8") as f:
    f.write(css)

print("CSS cleaned successfully.")
