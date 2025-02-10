import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import datetime
import numpy as np
import time
import face_recognition
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="PortrAid - Professional Portrait Analyzer",
    layout="wide"
)

# Constants from the paper
COMPOSITION_WEIGHTS = {
    'rule_of_thirds': 0.35,
    'eye_line': 0.25,
    'leading_space': 0.20,
    'headroom': 0.15,
    'background_balance': 0.05
}


# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input with enhanced detection"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


class CompositionAnalyzer:
    """Implements the paper's composition analysis techniques"""

    def __init__(self):
        self.weights = COMPOSITION_WEIGHTS

    def analyze_rule_of_thirds(self, face_location, image_shape):
        """Calculate Rule of Thirds score (Section 2.1 in paper)"""
        h, w = image_shape[:2]
        face_center = (
            (face_location[1] + face_location[3]) / 2,
            (face_location[0] + face_location[2]) / 2
        )

        # Calculate distances to third-line intersections
        thirds_x = [w / 3, 2 * w / 3]
        thirds_y = [h / 3, 2 * h / 3]

        min_dist = float('inf')
        for x in thirds_x:
            for y in thirds_y:
                dist = np.sqrt((face_center[0] - x) ** 2 + (face_center[1] - y) ** 2)
                min_dist = min(min_dist, dist)

        max_dist = np.sqrt(w ** 2 + h ** 2) / 2
        return 1 - (min_dist / (max_dist * 2))

    def analyze_eye_line(self, face_landmarks, image_shape):
        """Calculate Eye Line score (Section 2.2 in paper)"""
        if not face_landmarks:
            return 0.5

        h = image_shape[0]
        eye_points = face_landmarks['left_eye'] + face_landmarks['right_eye']
        eye_y = np.mean([p[1] for p in eye_points])
        ideal_eye_y = h / 3

        return 1 - abs(eye_y - ideal_eye_y) / (h / 2)

    def analyze_leading_space(self, face_landmarks, image_shape):
        """Calculate Leading Space score (Section 2.3 in paper)"""
        if not face_landmarks:
            return 0.5

        w = image_shape[1]
        face_direction = self._detect_face_direction(face_landmarks)
        nose_tip = face_landmarks['nose_bridge'][-1]

        space = w - nose_tip[0] if face_direction > 0 else nose_tip[0]
        return 1 - abs((space / w) - 0.35)

    def analyze_headroom(self, face_location, image_shape):
        """Calculate Headroom score (Section 2.4 in paper)"""
        h = image_shape[0]
        top_space = face_location[0]
        return 1 - abs(top_space / h - 0.2)

    def analyze_background_balance(self, image):
        """Calculate Background Balance score (Section 2.5 in paper)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        left_half = gray[:, :w // 2]
        right_half = gray[:, w // 2:]

        hist_left = cv2.calcHist([left_half], [0], None, [256], [0, 256])
        hist_right = cv2.calcHist([right_half], [0], None, [256], [0, 256])
        correlation = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)

        return (correlation + 1) / 2

    def _detect_face_direction(self, face_landmarks):
        """Detect face direction using facial landmarks"""
        left_eye = np.mean(face_landmarks['left_eye'], axis=0)
        right_eye = np.mean(face_landmarks['right_eye'], axis=0)
        nose_tip = face_landmarks['nose_bridge'][-1]

        eye_center = (left_eye + right_eye) / 2
        return np.sign(nose_tip[0] - eye_center[0])

    def get_overall_score(self, scores):
        """Calculate weighted overall score using Table 1 weights"""
        return sum(score * self.weights[metric] for metric, score in scores.items())


class EnhancedMultiScaleResNet(nn.Module):
    """Implementation of the multi-scale architecture from Section 3"""

    def __init__(self):
        super().__init__()
        self._setup_encoders()
        self._setup_attention()
        self._setup_classifier()
        self.crop_detector = self._setup_crop_detector()
        self.activation_maps = {}
        self.gradients = {}
        self._register_hooks()

    def _setup_encoders(self):
        """Setup the three encoders with shared architecture"""

        def create_encoder():
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256)
            )
            return model

        self.micro_encoder = create_encoder()
        self.meso_encoder = create_encoder()
        self.macro_encoder = create_encoder()

    def _modify_resnet(self, base_model):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        return model

    def _setup_attention(self):
        self.attention = nn.ModuleDict({
            'micro': self._create_attention_block(),
            'meso': self._create_attention_block(),
            'macro': self._create_attention_block()
        })

    def _create_attention_block(self):
        return nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _setup_classifier(self):
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def _setup_crop_detector(self):
        """Setup DeepCropDetect as described in Section 3.1"""
        detector = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        detector.fc = nn.Linear(detector.fc.in_features, 2)
        return detector

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activation_maps[name] = output

            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]

            return hook

        for name, encoder in [('micro', self.micro_encoder),
                              ('meso', self.meso_encoder),
                              ('macro', self.macro_encoder)]:
            encoder.layer4[-1].register_forward_hook(get_activation(f'{name}_layer4'))
            encoder.layer4[-1].register_full_backward_hook(get_gradient(f'{name}_layer4'))

    def forward(self, x):
        # Check for cropped image first
        crop_score = self.crop_detector(x)
        if torch.argmax(crop_score) == 1:
            return torch.tensor([[1.0, 0.0]])

        # Multi-scale analysis
        micro_features = self.micro_encoder(x)
        meso_features = self.meso_encoder(x)
        macro_features = self.macro_encoder(x)

        # Apply attention weights
        micro_weights = self.attention['micro'](micro_features)
        meso_weights = self.attention['meso'](meso_features)
        macro_weights = self.attention['macro'](macro_features)

        weighted_features = torch.cat([
            micro_features * micro_weights,
            meso_features * meso_weights,
            macro_features * macro_weights
        ], dim=1)

        return self.classifier(weighted_features)


def draw_composition_guides(frame, scores, face_landmarks):
    """Draw composition guides and metrics on frame"""
    height, width = frame.shape[:2]

    # Draw on-screen metrics FIRST (so they're always visible)
    y_pos = 25  # Start slightly lower from the top
    metrics = [
        ('Line', scores.get('eye_line', 0)),
        ('Headroom', scores.get('headroom', 0)),
        ('Rule of thirds', scores.get('rule_of_thirds', 0)),
        ('Eye position', scores.get('leading_space', 0))
    ]

    for label, value in metrics:
        text = f"{label}: {value:.1f}%"
        # Draw black outline for better visibility
        cv2.putText(frame, text, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 3)  # Thicker black outline
        # Draw white text
        cv2.putText(frame, text, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        y_pos += 20

    # Draw rule of thirds grid
    for i in range(1, 3):
        x = int(width * i / 3)
        y = int(height * i / 3)
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)

    if face_landmarks:
        # Draw eye line indicator
        left_eye = np.mean(face_landmarks['left_eye'], axis=0)
        right_eye = np.mean(face_landmarks['right_eye'], axis=0)
        cv2.line(frame,
                 tuple(map(int, left_eye)),
                 tuple(map(int, right_eye)),
                 (0, 255, 0), 2)

        # Draw face direction line
        nose_bridge = face_landmarks.get('nose_bridge', [])
        if nose_bridge:
            nose_tip = nose_bridge[-1]
            eye_center = (left_eye + right_eye) / 2
            # Draw diagonal line from eye center to nose tip
            cv2.line(frame,
                    tuple(map(int, eye_center)),
                    tuple(map(int, nose_tip)),
                    (0, 255, 255), 2)

    return frame

@st.cache_resource
def load_model():
    try:
        model = EnhancedMultiScaleResNet()
        checkpoint = torch.load('best_model(1).pth', map_location=torch.device('cpu'))

        # Create a new state dict with the correct keys
        new_state_dict = {}
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # First, collect all keys we want to transform
        keys_to_transform = list(state_dict.keys())  # Create a list of keys first

        # Map the old keys to the new model structure
        for key in keys_to_transform:
            value = state_dict[key]
            if key.startswith('conv1'):
                new_state_dict[f'micro_encoder.{key}'] = value
            elif key.startswith('bn1'):
                new_state_dict[f'micro_encoder.{key}'] = value
            elif key.startswith('layer'):
                new_state_dict[f'micro_encoder.{key}'] = value
            elif key.startswith('fc'):
                if '.1.' in key:
                    new_key = key.replace('.1.', '.0.')
                    new_state_dict[f'micro_encoder.{new_key}'] = value
                elif '.4.' in key:
                    new_key = key.replace('.4.', '.3.')
                    new_state_dict[f'micro_encoder.{new_key}'] = value
                elif '.7.' in key:
                    continue
                else:
                    new_state_dict[f'micro_encoder.{key}'] = value

        # Create a list of micro encoder keys before copying
        micro_keys = [k for k in new_state_dict.keys() if k.startswith('micro_encoder')]

        # Copy weights to meso and macro encoders
        for key in micro_keys:
            value = new_state_dict[key]
            meso_key = key.replace('micro_encoder', 'meso_encoder')
            macro_key = key.replace('micro_encoder', 'macro_encoder')
            new_state_dict[meso_key] = value.clone()
            new_state_dict[macro_key] = value.clone()

        # Initialize attention and classifier weights if they don't exist
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def analyze_frame(frame, model):
    """Analyze frame using both model predictions and composition rules"""
    try:
        # Convert frame for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and landmarks
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return None

        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
        if not face_landmarks:
            return None

        # Initialize analyzer
        analyzer = CompositionAnalyzer()

        try:
            # Get model prediction
            image = Image.fromarray(rgb_frame)
            image_tensor = preprocess_image(image)

            with torch.no_grad():
                model_output = model(image_tensor)
                model_scores = torch.softmax(model_output, dim=1)[0]
                model_composition_score = model_scores[1].item()  # Get positive class probability

            # Calculate geometry-based scores
            geometric_scores = {
                'rule_of_thirds': analyzer.analyze_rule_of_thirds(face_locations[0], frame.shape),
                'eye_line': analyzer.analyze_eye_line(face_landmarks[0], frame.shape),
                'leading_space': analyzer.analyze_leading_space(face_landmarks[0], frame.shape),
                'headroom': analyzer.analyze_headroom(face_locations[0], frame.shape),
                'background_balance': analyzer.analyze_background_balance(frame)
            }

            # Combine geometric scores with model prediction
            combined_scores = geometric_scores.copy()
            combined_scores['model_composition'] = model_composition_score

            # Weight the final scores (you can adjust these weights)
            model_weight = 0.4
            geometric_weight = 0.6

            # Calculate final scores
            final_scores = {}
            for metric, score in geometric_scores.items():
                final_scores[metric] = (score * geometric_weight + model_composition_score * model_weight) * 100

            # Add overall score
            final_scores['overall_score'] = analyzer.get_overall_score(geometric_scores) * geometric_weight + \
                                            model_composition_score * model_weight * 100

            return (final_scores, face_landmarks[0])

        except Exception as e:
            st.error(f"Score calculation error: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Frame analysis error: {str(e)}")
        return None


def main():
    st.title("PortrAid - Professional Portrait Analyzer")
    st.write("Real-time portrait composition analysis based on professional photography principles")

    # Load the model first
    model = load_model()
    if model is None:
        st.error("Failed to load model")
        return

    # Create layout
    col1, col2 = st.columns([3, 1])

    # Create empty containers
    with col2:
        analysis_placeholder = st.empty()

    with col1:
        st.subheader("Camera Feed")
        camera_placeholder = st.empty()

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("Failed to access webcam")
            return

        # Control buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Toggle Analysis"):
                st.session_state.analysis_active = not st.session_state.analysis_active
        with btn_col2:
            stop_btn = st.button("Stop")

        # Initialize analysis state if not present
        if 'analysis_active' not in st.session_state:
            st.session_state.analysis_active = True

        last_analysis_time = time.time()
        analysis_interval = 0.2  # Analyze every 200ms

        while not stop_btn:
            ret, frame = cap.read()
            if ret:
                current_time = time.time()

                # Only run analysis if active and enough time has passed
                if st.session_state.analysis_active and (current_time - last_analysis_time >= analysis_interval):
                    try:
                        # Pass model to analyze_frame
                        result = analyze_frame(frame, model)
                        last_analysis_time = current_time

                        if result is not None:
                            scores, face_landmarks = result

                            # Draw composition guides on frame
                            frame_with_guides = draw_composition_guides(frame.copy(), scores, face_landmarks)

                            # Show frame with guides
                            camera_placeholder.image(
                                cv2.cvtColor(frame_with_guides, cv2.COLOR_BGR2RGB),
                                channels="RGB",
                                use_column_width=True
                            )

                            # Update the fixed analysis container
                            with analysis_placeholder.container():
                                st.subheader("Live Analysis")
                                for metric, score in scores.items():
                                    if metric != 'model_composition':
                                        st.markdown(f"**{metric.replace('_', ' ').title()}**")

                                        if score >= 90:
                                            bar_color = 'green'
                                            message = "Excellent!"
                                        elif score >= 75:
                                            bar_color = 'blue'
                                            message = "Good"
                                        elif score >= 60:
                                            bar_color = 'orange'
                                            message = "Fair"
                                        else:
                                            bar_color = 'red'
                                            message = "Needs Improvement"

                                        st.progress(score / 100)
                                        st.markdown(
                                            f"<span style='color: {bar_color}'>{score:.1f}% - {message}</span>",
                                            unsafe_allow_html=True
                                        )
                        else:
                            # Draw helper grid on frame when no face is detected
                            height, width = frame.shape[:2]
                            frame_copy = frame.copy()

                            # Draw horizontal lines
                            for i in range(3):
                                y = int(height * (i + 1) / 4)
                                cv2.line(frame_copy, (0, y), (width, y), (0, 255, 0), 1)
                            # Draw vertical lines
                            for i in range(3):
                                x = int(width * (i + 1) / 4)
                                cv2.line(frame_copy, (x, 0), (x, height), (0, 255, 0), 1)

                            # Show frame with grid
                            camera_placeholder.image(
                                cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB),
                                channels="RGB",
                                use_column_width=True
                            )

                            # Update warning in the fixed container
                            with analysis_placeholder.container():
                                st.warning("""
                                No face detected. Please:
                                1. Ensure good lighting
                                2. Face the camera directly
                                3. Move closer to the camera
                                """)
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                else:
                    # Just show current frame with basic grid when not analyzing
                    height, width = frame.shape[:2]
                    frame_copy = frame.copy()
                    # Draw horizontal lines
                    for i in range(3):
                        y = int(height * (i + 1) / 4)
                        cv2.line(frame_copy, (0, y), (width, y), (0, 255, 0), 1)
                    # Draw vertical lines
                    for i in range(3):
                        x = int(width * (i + 1) / 4)
                        cv2.line(frame_copy, (x, 0), (x, height), (0, 255, 0), 1)

                    camera_placeholder.image(
                        cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_column_width=True
                    )

                time.sleep(0.01)

        cap.release()

if __name__ == "__main__":
    main()