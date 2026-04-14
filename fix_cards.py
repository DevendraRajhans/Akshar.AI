filepath = r'd:\AKSHAR_AI\Akshar.AI\app.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start and end markers
start_marker = "    # Display Categories as Premium Cards"
end_marker = "    active_cat_key = st.session_state.get"

start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if start_marker in line and start_idx is None:
        start_idx = i
    if end_marker in line and start_idx is not None:
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    print(f"Found block: lines {start_idx+1} to {end_idx}")
    
    new_block = '''    # Display Categories as Premium Cards
    cols = st.columns(3, gap="medium")
    for idx, cat in enumerate(learn_categories):
        with cols[idx]:
            is_active = st.session_state.get("learn_active_cat") == cat["key"]
            cat_cls = "premium-cat-card active" if is_active else "premium-cat-card"
            icon_svg = icon(cat["icon_name"], 28)
            cat_html = f\'\'\'
            <div class="{cat_cls}">
                <div class="cat-icon-container">{icon_svg}</div>
                <div class="cat-title">{cat["title"]}</div>
                <div class="cat-progress-text">{cat["learned"]}/{cat["total"]} Learned</div>
                <div class="cat-progress-bar">
                    <div class="cat-progress-fill" style="width: {(cat["learned"]/max(1, cat["total"]))*100}%"></div>
                </div>
            </div>
            \'\'\'
            st.markdown(cat_html, unsafe_allow_html=True)
            if st.button("Explore", key=f"btn_cat_{cat['key']}", use_container_width=True):
                if st.session_state.get("learn_active_cat") == cat["key"]:
                    st.session_state["learn_active_cat"] = None
                else:
                    st.session_state["learn_active_cat"] = cat["key"]
                st.rerun()

'''
    
    # Replace the block
    new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print('Block replaced successfully - reverted to original working code')
else:
    print(f'Markers not found: start={start_idx}, end={end_idx}')
