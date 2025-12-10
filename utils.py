import base64
from io import BytesIO
import os
import tempfile
from config import ADCB_LOGO_PATH, WATERMARK_IMAGE_PATH
import pyttsx3
from gtts import gTTS

# Import configuration
try:
    from config import WATERMARK_IMAGE_PATH
except ImportError:
    WATERMARK_IMAGE_PATH = None

# Handle optional dependencies gracefully
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Mock streamlit for testing
    class MockStreamlit:
        def markdown(self, *args, **kwargs): pass
        def audio(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def success(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def spinner(self, *args, **kwargs): 
            class MockSpinner:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return MockSpinner()
    st = MockStreamlit()

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


def img_to_base64(image_path):
    """Convert image to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        # This error is handled where the function is called, to avoid stopping the app
        return None





# -------------------- TTS HELPER FUNCTION --------------------
def play_text_as_speech(text_to_speak):
    """Generates speech from text and plays it using st.audio with preference for male voice."""
    try:
        # Try pyttsx3 first for local male voice
        import pyttsx3
        import tempfile
        import os
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Get all available voices
        voices = engine.getProperty('voices')
        
        # Look for male voices - more comprehensive search
        male_voice = None
        male_keywords = ['david', 'james', 'mark', 'mike', 'john', 'peter', 'steve', 'chris', 'alex', 'sam']
        
        # First, try exact matches for common male names
        for voice in voices:
            voice_name_lower = voice.name.lower()
            for keyword in male_keywords:
                if keyword in voice_name_lower:
                    male_voice = voice
                    break
            if male_voice:
                break
        
        # If no male voice found, try looking for 'male' in the name
        if not male_voice:
            for voice in voices:
                if 'male' in voice.name.lower():
                    male_voice = voice
                    break
        
        # If still no male voice, use the first available voice
        if not male_voice and voices:
            male_voice = voices[0]
        
        # Set the voice
        if male_voice:
            engine.setProperty('voice', male_voice.id)
        
        # Set speech rate and volume
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech and save to file
        engine.save_to_file(text_to_speak, temp_filename)
        engine.runAndWait()
        
        # Read the file and play it
        with open(temp_filename, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        # Play audio
        st.audio(audio_data, format='audio/wav', autoplay=True)
        
    except ImportError:
        # Fallback to gTTS if pyttsx3 is not available
        try:
            # Use a specific language and TLD that might provide a male voice
            tts = gTTS(text=text_to_speak, lang='en', tld='com.au', slow=False)
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format='audio/mp3', autoplay=True)
        except Exception as e:
            st.warning(f"Could not play speech: {e}")
    except Exception as e:
        # Fallback to gTTS if pyttsx3 fails
        try:
            tts = gTTS(text=text_to_speak, lang='en', tld='com.au', slow=False)
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format='audio/mp3', autoplay=True)
        except Exception as e2:
            st.warning(f"Could not play speech: {e2}")


def test_tts_voices():
    """Test function to debug TTS voice selection - call this to see available voices"""
    try:
        import pyttsx3
        
        st.markdown("## 🧪 TTS Voice Test")
        st.info("Testing available voices on your system...")
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Get all available voices
        voices = engine.getProperty('voices')
        
        st.success(f"✅ Found {len(voices)} available voices:")
        
        # Display all voices
        for i, voice in enumerate(voices):
            voice_type = "🎤 Male" if any(keyword in voice.name.lower() for keyword in ['david', 'james', 'mark', 'mike', 'john', 'peter', 'steve', 'chris', 'alex', 'sam', 'male']) else "👩 Female"
            st.write(f"  {i}: {voice.name} | ID: {voice.id} | {voice_type}")
        
        # Test each voice
        st.markdown("### 🔊 Test Voices")
        test_text = "Hello, this is a test of the text to speech functionality."
        
        for i, voice in enumerate(voices):
            if st.button(f"Test Voice {i}: {voice.name}", key=f"test_voice_{i}"):
                try:
                    # Set the voice
                    engine.setProperty('voice', voice.id)
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 0.9)
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_filename = temp_file.name
                    
                    # Generate speech
                    engine.save_to_file(test_text, temp_filename)
                    engine.runAndWait()
                    
                    # Read and play
                    with open(temp_filename, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    
                    # Clean up
                    os.unlink(temp_filename)
                    
                    # Play audio
                    st.audio(audio_data, format='audio/wav', autoplay=True)
                    st.success(f"✅ Playing voice: {voice.name}")
                    
                except Exception as e:
                    st.error(f"❌ Failed to test voice {voice.name}: {str(e)}")
        
    except ImportError:
        st.error("❌ pyttsx3 not installed. Install with: pip install pyttsx3")
    except Exception as e:
        st.error(f"❌ TTS test failed: {str(e)}")


def apply_custom_css():
    """Applies custom CSS for styling, including sidebar watermark and mobile responsiveness."""
    watermark_base64 = img_to_base64(WATERMARK_IMAGE_PATH)
    
    # Base CSS for titles and sidebar width with mobile responsiveness
    st.markdown(f"""
    <style>
    
        /* --- Sidebar Font Size --- */
        [data-testid="stSidebar"] * {{
            font-size: 1.2rem; /* Adjust this value to your liking */
        }}
        
        /* Main Title Styles - Responsive */
        .main-title {{
            font-size: clamp(2rem, 8vw, 6rem) !important;
            font-weight: 700;
            color: #e4002b; /* ADCB Red */
            text-align: left;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }}
        /* Login Title Styles - Responsive */
        .login-title {{
            color: #e4002b !important;
            font-size: clamp(1.5rem, 6vw, 3rem) !important;
            font-weight: 700 !important;
            margin-bottom: 2rem !important;
        }}
        /* Additional Login Title Style to ensure color override */
        [data-testid="stMarkdownContainer"] h1 {{
            color: #e4002b !important;
        }}
        /* Welcome Image Styles */
        .welcome-image {{
            max-width: 200px;
            width: 90%;
            height: auto;
            margin: 2rem auto;
            display: block;
        }}
        .sub-title {{
            font-size: clamp(1.2rem, 4vw, 2rem);
            color: #333;
            text-align: left;
            margin-bottom: 20px;
            line-height: 1.4;
        }}
        
        /* --- New Sidebar Callout Style --- */
        .sidebar-callout {{
            padding: 1rem 1rem 1rem 1.5rem; /* More padding on the left for the border */
            margin: 1rem;
            background-color: #f8f9fa;
            border-left: 5px solid #cccccc; /* Grey edge */
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
        }}
        .sidebar-callout:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }}
        .sidebar-callout p {{
            color: #e4002b; /* ADCB Red text */
            font-size: 1.2rem; /* Slightly larger font */
            font-weight: 700; /* Bold */
            font-style: italic; /* Italic */
            line-height: 1.5;
            margin: 0;
            text-align: justify;
            text-indent: 2em;
        }}
        
        /* Responsive Sidebar */
        [data-testid="stSidebar"] {{
            width: 100% !important;
            max-width: 450px !important;
        }}
        
        /* Mobile-specific adjustments */
        @media (max-width: 768px) {{
            [data-testid="stSidebar"] {{
                width: 100% !important;
                max-width: 100% !important;
            }}
            
            /* Make charts responsive */
            .js-plotly-plot {{
                width: 100% !important;
                height: auto !important;
            }}
            
            /* Adjust font sizes for mobile */
            [data-testid="stSidebar"] * {{
                font-size: 1rem !important;
            }}
            
            /* Make inputs more touch-friendly */
            .stSlider > div {{
                min-height: 60px !important;
            }}
            
            .stButton > button {{
                min-height: 50px !important;
                font-size: 1.1rem !important;
            }}
            
            /* Stack columns on mobile */
            [data-testid="column"] {{
                width: 100% !important;
                margin-bottom: 1rem !important;
            }}
        }}
        
        /* Custom Expander Styles for 'What-If' */
        [data-testid="stExpander"] summary {{
            font-size: clamp(1rem, 3vw, 1.5rem) !important;
            font-weight: bold !important;
            padding: 1rem 0 !important;
        }}
        
        /* Make cost breakdown cards mobile-friendly */
        @media (max-width: 768px) {{
            div[style*="font-family: 'Consolas'"] {{
                font-size: 0.9rem !important;
                padding: 20px !important;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Add watermark if image is available
    if watermark_base64:
        st.markdown(f"""
        <style>
            [data-testid="stSidebar"] > div:first-child {{
                position: relative;
            }}
            [data-testid="stSidebar"] > div:first-child::before {{
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: url("data:image/png;base64,{watermark_base64}");
                background-size: 70%;
                background-repeat: no-repeat;
                background-position: center 20px;
                opacity: 0;
                z-index: -1;
                pointer-events: none;
            }}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Show a warning in the sidebar if the watermark image is not found
        st.sidebar.warning("Sidebar watermark image not found.", icon="⚠️")


def create_breakdown_chart(results, user_data, tx, tx_cost):
    """Creates a stacked bar chart to show the breakdown of costs for each package."""
    breakdown_data = []

    # 1. No package breakdown
    no_pkg_breakdown = {
        "Transactions (Int'l, Dom, Chq)": (tx['international'] * tx_cost['international'] +
                                           tx['domestic'] * tx_cost['domestic'] +
                                           tx['cheque'] * tx_cost['cheque']),
        "Services (PDC, Inward FCY)": (user_data['pdc_count'] * user_data['pdc_cost'] +
                                       user_data['inward_fcy_count'] * user_data['inward_fcy_cost']),
        "WPS/CST Cost": user_data['wps_cost'],
        "Other Costs": user_data['other_costs_input'],
    }
    if user_data['fx_amount'] > 0:
        fx_impact_no_pkg = user_data['fx_amount'] * user_data['client_fx_rate']
        if user_data['fx_direction'] == "Sell USD":
             fx_impact_no_pkg = -fx_impact_no_pkg
        no_pkg_breakdown["FX Impact"] = fx_impact_no_pkg
    
    for component, cost in no_pkg_breakdown.items():
        if cost != 0:
             breakdown_data.append({"Category": "Without Package", "Cost Component": component, "Cost (AED)": cost})

    # 2. Package breakdowns from results
    for name, result_details in results.items():
        breakdown = result_details['breakdown']
        
        # Group components for a cleaner chart legend
        grouped_breakdown = {}
        if breakdown.get("Package Cost", 0) != 0:
            grouped_breakdown["Package Fee"] = breakdown.get("Package Cost", 0)
        
        txn_cost = (breakdown.get("International Transactions Cost", 0) + 
                    breakdown.get("Domestic Transactions Cost", 0) + 
                    breakdown.get("Cheque Transactions Cost", 0))
        if txn_cost != 0: grouped_breakdown["Transactions (Paid)"] = txn_cost
        
        services_cost = (breakdown.get("Pdc Cost", 0) + 
                         breakdown.get("Inward Fcy Remittance Cost", 0))
        if services_cost != 0: grouped_breakdown["Services (Paid)"] = services_cost
        
        if breakdown.get("FX Impact", 0) != 0:
            grouped_breakdown["FX Impact"] = breakdown["FX Impact"]
        
        if breakdown.get("Other Costs (User Input)", 0) != 0:
            grouped_breakdown["Other Costs"] = breakdown["Other Costs (User Input)"]
        
        for component, cost in grouped_breakdown.items():
            breakdown_data.append({"Category": name, "Cost Component": component, "Cost (AED)": cost})

    if not breakdown_data:
        return None

    df_breakdown = pd.DataFrame(breakdown_data)
    fig_breakdown = px.bar(df_breakdown, x="Category", y="Cost (AED)", color="Cost Component", 
                           title="📊 Detailed Cost Breakdown by Component",
                           labels={"Cost (AED)": "Cost (AED)", "Category": "Option", "Cost Component": "Component"},
                           text_auto=',.0f')
    fig_breakdown.update_layout(bargap=0.3, legend_title_text='Cost Component')
    return fig_breakdown


# --- Authentication Logic ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False
            st.error("Incorrect password. Please try again.")

    # Initialize session state
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        
    # First run, show input for password
    if not st.session_state["password_correct"]:
        # Convert logo to base64
        adcb_logo_base64 = img_to_base64(ADCB_LOGO_PATH)
        
        # Use direct title with custom styling and ADCB logo
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 1rem;">
                <h1 style="color: #e4002b; font-size: 4rem; font-weight: 700; margin: 0;">
                    Fikra Genie 🧞‍♂️💡 - Login
                </h1>
                <img src="data:image/png;base64,{adcb_logo_base64}" style="height: 80px; object-fit: contain;" alt="ADCB Logo">
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <h2 style="font-size: 2rem; margin-bottom: 1rem;">
                Please enter the password to access the application.
            </h2>
        """, unsafe_allow_html=True)
        
        # Password input and login button
        password = st.text_input("Password", type="password", key="password")
        if st.button("Login", type="primary"):
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        return False
    
    return True

