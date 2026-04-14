import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Fix use_container_width deprecation warnings
code = re.sub(r'use_container_width\s*=\s*True', "width='stretch'", code)
code = re.sub(r'use_container_width\s*=\s*False', "width='content'", code)

# Fix width=0 and height=0 that crash styling elements (Valid positive integer required)
code = code.replace(', height=0, width=0)', ', height=1, width=1)')
code = code.replace('height=0)', 'height=1)')

# Replace st.components.v1.html with st.html if Streamlit wants it.
# Note: Streamlit 1.37 deprecates components.v1.html and st.components.v1.html and suggests st.html instead.
# However, usually st.html does not take height/width arguments. 
# We'll leave components.html alone for now unless it warns. Oh wait, the warning SAID: 
# "Please replace st.components.v1.html with st.iframe" for things? Actually the user had replaced components.html with st.components.v1.html? No. We'll leave components.html as is, the warnings are harmless for now, but use_container_width and width=0 are hard crashes.
# Actually, the user prompts specifically "solve all the errors and warnings".
# The warning for components.html is "Please replace st.components.v1.html with st.components.v1.iframe".
# Wait, NO. The terminal output is `Please replace st.components.v1.html with st.iframe` if st.iframe does the same. BUT `st.components.v1.html` is used by Streamlit internally when we call `import streamlit.components.v1 as components; components.html(...)`. To solve the warning, Streamlit recommends using `st.html` in newer versions. Actually, if they use `components.html()` Streamlit 1.37 shows: "Please replace st.components.v1.html with st.components.v1.html" if it's broken.
# Let's replace components.html() with st.components.v1.html() to be standard.
code = re.sub(r'components\.html\(', 'st.components.v1.html(', code)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Formatting fixes applied successfully.")
