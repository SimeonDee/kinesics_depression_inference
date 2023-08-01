import streamlit as st


def load_styles():
    return """
        <style>
            .details-wrapper{
                margin: 20px 55px;
            }

            .about-app{
                margin-top: 25px;
                padding: 20px;
                padding-left: 0;
                font-size: 3rem;
                word-spacing: 2px;
                letter-spacing: 2px;
            }

            .footer{
                margin-top: 30px;
                padding: 5px 10px;
                background-color: #444;
                color: #ccc;
                text-align: center;
            }

            .value{
                text-transform: uppercase;
            }

            .label{
                font-weight: bold;
            }

            td{
                padding: 5px 10px;
                font-size: large;

                border: 1px solid rgb(95, 81, 81);
                border-radius: 5px;
            }

            .input_div_wrapper, .input_div_wrapper2{
                padding:20px;
                min-height: 200px;
                border: 1px solid #555;
                border-radius: 10px;
            }

            .input_div_wrapper2{
                min-height: 320px;
            }

            h3{
                padding-bottom: 0px;
            }

            .final-prediction{
                font-size: xx-large;
                font-weight: bold;
                display: inline;
            }

            .final-prediction-confidence{
                background-color: #ccc;
                padding: 10px;
                border-radius: 10px;
                width: fit-content;
                color: #444;
                font-weight: bold;
            }

            .confidence-wrapper{
                border-top: 1px solid #ccc;
                padding-top: 5px;
            }

            .final-wrapper{
                padding: 20px 50px;
                border: 1px solid #888;
                border-radius: 15px;
                width: 100%;
                background-color: #444;
                color: #ccc;
            }
            
            .final-wrapper h4{
                color: #eee;
            }

            .entry-msg-wrapper{
                padding: 10px 20px;
                margin-top: 20px;
                margin-bottom: 0px;
                border: 1px solid #888;
                border-radius: 5px;
                width: 100%;
                background-color: #444;
                color: #ccc;
                text-align: center;
            }

        </style>
    """


def show_details():
    return """
        <div class="details-wrapper">
            <table>
                <tr>
                    <td class="label">Name:</td>
                    <td class="value">Awoyelu Tolulope Moyo</td>
                </tr>
                <tr>
                    <td class="label">Matric:</td>
                    <td class="value">TP18/19/H/0929</td>
                </tr>
                <tr>
                    <td class="label">Degree:</td>
                    <td class="value">Ph.D (Intelligent Systems Engineering)</td>
                </tr>
                <tr>
                    <td class="label">Institution:</td>
                    <td class="value">Obafemi Awolowo University, Ile-Ife, Nigeria</td>
                </tr>
                <tr>
                    <td class="label">Date:</td>
                    <td class="value">August, 2023</td>
                </tr>
                <tr>
                    <td class="label">Supervisor By:</td>
                    <td class="value">Dr. A. R. Iyanda</td>
                </tr>
            </table>
        </div>
    """


# --- Loading css styles ---
st.write(load_styles(), unsafe_allow_html=True)
st.write('# Development of a Multimodal System for Detection of Depressive Symptoms from Images and Textual Posts')
st.write(show_details(), unsafe_allow_html=True)

st.write("""
    #
    <h3> About App </h3>
    <div class="about-app">
         <p> This app classifies social media text, image or both as depressive or not using SVC and CNN algorithms.</p>
    </div>
         
    <div class="footer">
        <p> &copy; August 2023 </p>
    </div>
""", unsafe_allow_html=True)
