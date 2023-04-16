import streamlit as st

if 'context' not in st.session_state: st.session_state['context'] = dict()

if __name__ == '__main__':
    with st.form("form", clear_on_submit=True):
        key = st.text_input("key")
        context = st.text_area("Some context here")
        if st.form_submit_button("Submit"): st.session_state['context'][key] = context
    st.write(st.session_state['context'])
