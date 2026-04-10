import streamlit as st
import streamlit.components.v1 as components

st.title("Confetti Test")

if st.button("Pop"):
    components.html("""
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
        try {
            const parentDoc = window.parent.document;
            const canvas = parentDoc.createElement('canvas');
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.pointerEvents = 'none';
            canvas.style.zIndex = '999999';
            parentDoc.body.appendChild(canvas);

            var myConfetti = confetti.create(canvas, {
                resize: true,
                useWorker: true
            });

            myConfetti({
                particleCount: 150,
                spread: 100,
                origin: { y: 0.6 }
            });

            setTimeout(() => {
                parentDoc.body.removeChild(canvas);
            }, 3500);
        } catch(e) {
            console.error(e);
        }
    </script>
    """, height=0, width=0)
