'''AI Document Summarizer - Streamlit + Google Gemini.'''

import os
import re
import uuid
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title='AI Summarizer', page_icon='📝', layout='centered')

NL = '\n'
MAX_FILE_MB = 15
MAX_TOTAL_MB = 18
SYSTEM_INSTRUCTION = (
    'You are a careful, accurate document summarizer. Follow the requested format '
    'exactly and never invent facts.'
)
ALLOWED_EXTS = ['pdf', 'png', 'jpg', 'jpeg', 'webp']
MIME_BY_EXT = {
    'pdf': 'application/pdf',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'webp': 'image/webp',
}
MODELS = ['gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3.5-flash']

STYLES = {
    'Bullet Points': (
        'Write clear bullet points. Group related points under short bold headings. '
        'Each bullet should hold one idea in under 25 words.'
    ),
    'Paragraph': (
        'Write flowing paragraphs in plain prose. Do not use bullet points or headings.'
    ),
    'Comprehension': (
        'Write a study-style summary with these sections in order, using bold headings: '
        'Overview (2-3 sentences), Key Points (bullets), Important Terms (term: short meaning), '
        'and Comprehension Questions (5 questions, each followed by a short answer).'
    ),
    'Quick TL;DR': (
        'Write one bold sentence that captures the whole document, then 3 to 5 bullets with '
        'the most important details, then a final line starting with Action items: '
        'listing anything the reader must do (write None if there is nothing).'
    ),
}

LENGTHS = {
    'Short': 'about 100 words',
    'Medium': 'about 250 words',
    'Detailed': 'about 500 words',
}

LANGUAGES = [
    'Same as document',
    'English',
    'Hindi',
    'Hinglish (Hindi written in English letters)',
    'Urdu',
    'Spanish',
    'French',
    'German',
    'Arabic',
]


def get_api_key():
    try:
        key = st.secrets['GEMINI_API_KEY']
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get('GEMINI_API_KEY', '')


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    return genai.Client(api_key=api_key)


def detect_mime(name, declared):
    ext = name.rsplit('.', 1)[-1].lower() if name and '.' in name else ''
    if ext in MIME_BY_EXT:
        return MIME_BY_EXT[ext]
    if declared in MIME_BY_EXT.values():
        return declared
    return None


def collect_items(uploaded_files, camera_photo):
    items = []
    problems = []
    for f in uploaded_files or []:
        data = f.getvalue()
        mime = detect_mime(f.name, getattr(f, 'type', None))
        if mime is None:
            problems.append(f'{f.name}: unsupported file type.')
        elif len(data) == 0:
            problems.append(f'{f.name}: the file is empty.')
        elif len(data) > MAX_FILE_MB * 1024 * 1024:
            problems.append(f'{f.name}: larger than {MAX_FILE_MB} MB.')
        else:
            items.append({'name': f.name, 'mime': mime, 'data': data})
    if camera_photo is not None:
        data = camera_photo.getvalue()
        if data:
            mime = detect_mime(getattr(camera_photo, 'name', ''), getattr(camera_photo, 'type', None))
            items.append({'name': 'Camera photo', 'mime': mime or 'image/jpeg', 'data': data})
    return items, problems


def build_prompt(style, length, language, focus, combined):
    if language == 'Same as document':
        lang_rule = 'Write in the same language as the document.'
    else:
        lang_rule = f'Write the summary in {language}.'
    if combined:
        source = 'the attached documents, treated as one combined source'
    else:
        source = 'the attached document'
    focus_rule = f'Pay special attention to: {focus.strip()}.' if focus.strip() else ''
    return f'''You are an expert summarizer. Summarize {source}.

FORMAT: {STYLES[style]}
LENGTH: {LENGTHS[length]} (the Comprehension style may run a little longer).
LANGUAGE: {lang_rule}
{focus_rule}

RULES:
- Use only information that appears in the document. Never invent facts.
- Keep names, numbers and dates exact.
- If part of the document is unreadable, say so briefly instead of guessing.
- Output only the summary, with no preamble such as Here is the summary.'''


def summarize(client, model, items, prompt):
    parts = [types.Part.from_bytes(data=i['data'], mime_type=i['mime']) for i in items]
    response = client.models.generate_content(
        model=model,
        contents=parts + [prompt],
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    text = (getattr(response, 'text', None) or '').strip()
    if not text:
        raise ValueError(
            'The model returned no text. The file may be unreadable or blocked by '
            'safety filters. Try another file or model.'
        )
    return text


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if 'api key' in low or 'api_key' in low or '401' in low or '403' in low or 'permission_denied' in low:
        return 'API key problem. Check that your Gemini key is correct and active.'
    if '429' in low or 'quota' in low or 'resource_exhausted' in low:
        return 'Rate limit or quota reached. Wait a minute, or try another model in the sidebar.'
    if '404' in low or 'not found' in low:
        return 'That model is not available. Pick a different model in the sidebar.'
    if '503' in low or '500' in low or 'unavailable' in low or 'overloaded' in low:
        return 'The AI service is busy right now. Please try again shortly.'
    return f'Something went wrong: {msg}'


def run_summaries(client, model, items, style, length, language, focus, combined):
    jobs = [items] if combined else [[i] for i in items]
    results = []
    progress = st.progress(0.0, text='Starting...')
    for n, group in enumerate(jobs, start=1):
        if combined:
            label = 'Combined: ' + ', '.join(i['name'] for i in group)
        else:
            label = group[0]['name']
        progress.progress((n - 1) / len(jobs), text=f'Summarizing {n} of {len(jobs)}...')
        try:
            prompt = build_prompt(style, length, language, focus, combined)
            text = summarize(client, model, group, prompt)
            results.append({
                'id': uuid.uuid4().hex[:8],
                'name': label,
                'style': style,
                'length': length,
                'model': model,
                'time': datetime.now().strftime('%d %b %Y, %H:%M'),
                'text': text,
            })
        except Exception as exc:
            st.error(f'{label}: {friendly_error(exc)}')
    progress.empty()
    return results


def safe_filename(name):
    base = name.rsplit('.', 1)[0] if '.' in name else name
    base = re.sub(r'[^A-Za-z0-9_-]+', '_', base).strip('_')[:40]
    return (base or 'summary') + '_summary.md'


def build_export(results):
    lines = [f'# Summaries ({datetime.now().strftime("%d %b %Y, %H:%M")})', '']
    for r in results:
        lines += [f'## {r["name"]} ({r["style"]}, {r["length"]})', '', r['text'], '']
    return NL.join(lines)


def show_results(results):
    if not results:
        return
    st.divider()
    st.subheader('Your summaries')
    for idx, r in enumerate(results):
        with st.expander(f'{r["name"]} · {r["style"]} · {r["length"]}', expanded=(idx == 0)):
            tab_view, tab_copy = st.tabs(['Summary', 'Copy text'])
            with tab_view:
                st.markdown(r['text'])
            with tab_copy:
                st.code(r['text'], language=None)
            st.caption(f'{len(r["text"].split())} words · {r["model"]} · {r["time"]}')
            st.download_button(
                'Download this summary (.md)',
                data=r['text'],
                file_name=safe_filename(r['name']),
                mime='text/markdown',
                key=f'dl_{r["id"]}',
            )
    if len(results) > 1:
        st.download_button(
            'Download all summaries (.md)',
            data=build_export(results),
            file_name='all_summaries.md',
            mime='text/markdown',
            key='dl_all',
        )


def main():
    if 'results' not in st.session_state:
        st.session_state.results = []

    st.title('📝 AI Document Summarizer')
    st.caption('Upload a PDF or image, choose a style, and get a clean summary.')

    with st.sidebar:
        st.header('Settings')
        api_key = get_api_key()
        if api_key:
            st.success('API key loaded securely.')
        else:
            st.error('API key missing. Add GEMINI_API_KEY in your app Secrets.')
        model = st.selectbox('Model', MODELS)
        language = st.selectbox('Output language', LANGUAGES)
        if st.button('Clear all summaries'):
            st.session_state.results = []
            st.rerun()

    tab_files, tab_camera = st.tabs(['📁 Upload', '📷 Camera'])
    with tab_files:
        uploaded = st.file_uploader(
            'PDF or image files', type=ALLOWED_EXTS, accept_multiple_files=True,
        )
    with tab_camera:
        photo = st.camera_input('Take a photo of a page')

    items, problems = collect_items(uploaded, photo)
    for p in problems:
        st.warning(p)
    if items:
        st.caption(f'{len(items)} file(s) ready.')

    col_a, col_b = st.columns(2)
    with col_a:
        style = st.radio('Summary style', list(STYLES.keys()))
    with col_b:
        length = st.select_slider('Length', options=list(LENGTHS.keys()), value='Medium')

    focus = st.text_input(
        'Focus on (optional)', placeholder='e.g. deadlines, key numbers, exam topics',
    )
    combined = False
    if len(items) > 1:
        combined = st.checkbox('Combine all files into one summary')

    if st.button('✨ Summarize', type='primary'):
        if not api_key:
            st.error('API key not found. Add GEMINI_API_KEY to your Streamlit Secrets.')
        elif not items:
            st.warning('Upload a file or take a photo first.')
        elif combined and sum(len(i['data']) for i in items) > MAX_TOTAL_MB * 1024 * 1024:
            st.error(f'Combined files are over {MAX_TOTAL_MB} MB. Summarize them separately.')
        else:
            try:
                client = get_client(api_key)
            except Exception as exc:
                st.error(friendly_error(exc))
            else:
                new_results = run_summaries(
                    client, model, items, style, length, language, focus, combined,
                )
                if new_results:
                    st.session_state.results = new_results + st.session_state.results
                    st.success('Done! Scroll down to read your summary.')

    show_results(st.session_state.results)


main()