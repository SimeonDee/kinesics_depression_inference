import streamlit as st
import kinesics
from kinesics import KinesicsHybridModel

from utils import (show_details, get_tag, load_styles,
                   visualize_predictions_in_bars, show_final_prediction,
                   show_entry_msg, get_image, get_tag_and_image)

# --- GENERAL SETTINGS ---
st.session_state['cur_image_path'] = None
post_tag = None
post_image = None

# --- PAGE SETTINGS ---
TITLE = 'Development of a Multimodal System for Detection of Depressive Symptoms from Images and Textual Posts'

st.set_page_config(page_title='Kinesics',
                   page_icon='depressed_icon.png', layout='centered')
st.write(load_styles(), unsafe_allow_html=True)


# --- Loading Trained model ---
@st.cache_resource
def load_trained_model(model_dir='models'):
    model = KinesicsHybridModel(model_dir_or_url=model_dir)
    return model


kinesics_model = load_trained_model('models')


# --- MAIN PAGE
st.write(f"## {TITLE}")
st.write(show_details(), unsafe_allow_html=True)

st.write("#")
st.write(show_entry_msg(), unsafe_allow_html=True)

st.write("---")
choices = ('--- select problem category ---',
           'Text/Tag Only', 'Image Only', 'Tag and Image')
problem_type = st.selectbox("Select Problem Type: ", choices)
st.write("---")

# --- IF ONLY TEXT ---
if problem_type.lower() == 'text/tag only':
    post_tag = get_tag(st)
    if post_tag is not None:
        post_image = None

        # TO-DO: prediction
        results = kinesics_model.predict_text_post(post_tag)

        st.write('#')
        st.write('### Text Classification Results')
        st.write('---')
        st.write(results)
        visualize_predictions_in_bars(st, prediction_results=results)
        st.write('---')

        st.markdown(show_final_prediction(
            predicted=results['prediction'], confidence=results['confidence']), unsafe_allow_html=True)

    else:
        st.error("No text supplied yet.")

elif problem_type.lower() == 'image only':
    post_image = get_image(st)
    if post_image is not None:
        post_tag = None

        # TO-DO: prediction
        results = kinesics_model.predict_single_image_with_details(
            img_path=st.session_state['cur_image_path'])

        st.write('#')
        st.write('### Image Classification Results')
        st.write('---')
        st.write(results)
        visualize_predictions_in_bars(st, prediction_results=results)
        st.write('---')

        st.markdown(show_final_prediction(
            predicted=results['prediction'], confidence=results['confidence']), unsafe_allow_html=True)

    else:
        st.error("No image data received.")

elif problem_type.lower() == 'tag and image':
    data = get_tag_and_image(st)
    if data is not None:
        post_tag, post_image = data

        # TO-DO: prediction
        text_results, image_results = kinesics_model.predict_hybrid(
            post=post_tag, img_path=st.session_state['cur_image_path'])

        avg_performance = kinesics_model.get_avg_prediction_with_details(
            text_results, image_results)

        st.write('##')
        st.write('### Post Classification Results')
        st.write('---')

        col1, col2 = st.columns([2, 2], gap='medium')
        with col1:
            st.write('##### Text Classification Results (SVC)')
            st.write('---')
            visualize_predictions_in_bars(st, prediction_results=text_results)
            st.write('---')

        with col2:
            st.write('##### Image Classification Results (CNN)')
            st.write('---')
            visualize_predictions_in_bars(st, prediction_results=image_results)
            st.write('---')

        col1, col2 = st.columns([1, 1], gap='medium')
        with col1:
            st.write(text_results)

        with col2:
            st.write(image_results)

        st.write('---')

        st.write('####')
        st.write('## Hybrid Model Result (SVC + CNN)')
        st.write('---')
        st.write(avg_performance)
        visualize_predictions_in_bars(st, prediction_results=avg_performance)
        st.write('---')

        st.markdown(show_final_prediction(
            predicted=avg_performance['prediction'], confidence=avg_performance['confidence']), unsafe_allow_html=True)

    else:
        st.error("No image data received.")
