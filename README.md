# Class Vision

**Intelligent Classroom Attendance and Presence Monitoring using Edge AI**

Class Vision is an edge-AI classroom monitoring project designed to support automated attendance through person tracking and face recognition. It is developed to run on an NVIDIA Jetson AGX Orin Developer Kit, keeping the core video-processing workflow close to the camera rather than depending entirely on cloud inference.

> **Project status:** Active development / prototype. Features, performance, deployment procedures, and supported hardware may change. Do not treat this repository as a production-ready attendance system without completing the security, privacy, reliability, and licensing work described below.

## Overview

Class Vision combines computer-vision components to support a classroom attendance workflow:

- **Person detection:** Detect people in the camera stream using YOLOv8.
- **Person tracking:** Associate detections across frames using tracking logic (ByteTrack is used in the current development workflow).
- **Face detection:** Detect faces using the SCRFD model distributed in the InsightFace `buffalo_l` model pack.
- **Face recognition:** Generate face embeddings using an ArcFace model and compare them with an enrolled face gallery.
- **Attendance workflow:** Record recognized student attendance in a local SQLite database.
- **Unknown-face review:** Provide a faculty review workflow for unknown or uncertain face observations.
- **Faculty review dashboard:** A Streamlit-based interface is included for reviewing attendance and recognition cases.

The project is intended to support a human-reviewed workflow. A face-recognition result should not be treated as unquestionable proof of identity or presence.

## Current technology stack

| Area | Technology |
|---|---|
| Edge hardware | NVIDIA Jetson AGX Orin Developer Kit |
| Operating environment | Ubuntu for Jetson / JetPack environment |
| Programming language | Python |
| Person detection | Ultralytics YOLOv8 (currently `yolov8n.pt`) |
| Person tracking | ByteTrack-based tracking workflow |
| Face detection | InsightFace SCRFD |
| Face recognition | InsightFace ArcFace |
| Deep-learning inference | PyTorch; ONNX Runtime for InsightFace models |
| Review dashboard | Streamlit |
| Local attendance storage | SQLite |
| Development tools | Git and GitHub |

Exact package versions and model/runtime compatibility depend on the JetPack installation. See the environment notes below and verify compatibility before changing packages on the Jetson.

## Repository structure

The repository contains the application scripts and dashboard components. The main files currently include:

```text
Class_vision/
├── attendai_attendance_session_unknown_queue.py
├── attendai_faculty_review.py
├── dashboard/
│   ├── components/
│   ├── dashboard-layout.tsx
│   ├── dashboard.css
│   ├── data.tsx
│   └── demo.tsx
├── config/
├── data/
├── tests and validation scripts
└── README.md
```

This is a high-level guide, not a complete directory listing. Development, diagnostic, and validation scripts may also be present.

## Hardware and software requirements

The current development target is:

- NVIDIA Jetson AGX Orin Developer Kit
- JetPack 5.1.3 / L4T R35.5.0 development environment
- Python 3.8
- CUDA-enabled PyTorch compatible with the installed JetPack release
- OpenCV
- Ultralytics
- InsightFace and a compatible ONNX Runtime installation
- A supported USB camera (the current test setup uses a Logitech UVC camera)
- Streamlit for the faculty review dashboard

**Important:** Jetson packages are hardware- and JetPack-specific. Do not install generic desktop CUDA/PyTorch/ONNX Runtime wheels without checking compatibility. The current development setup has had CUDA-enabled PyTorch available while InsightFace ONNX inference used CPU providers; verify the active execution providers and benchmark the full pipeline on the target device.

## Setup

The following commands reflect the current Jetson development layout. They assume the project is located at `~/Desktop/Indiran` and the AI environment already exists at `~/yolo_env`.

### 1. Get the source code

```bash
git clone git@github.com:indhiran08-coder/Class_vision.git
cd Class_vision
```

If you use HTTPS instead of SSH:

```bash
git clone https://github.com/indhiran08-coder/Class_vision.git
cd Class_vision
```

### 2. Activate the AI environment

If you have already created and configured the project environment:

```bash
source ~/yolo_env/bin/activate
```

Install dependencies only after checking compatibility with your JetPack and Python versions. A pinned, reproducible requirements file should be maintained before using this project on another device.

### 3. Prepare local data and models

The application requires an institution-approved student roster and an enrolled face gallery. These are intentionally not distributed in this repository.

Prepare the required local data using the project's enrollment/data-preparation workflow. Do not commit student records, student photographs, face embeddings, attendance databases, camera recordings, or other personal/biometric information to Git.

The person-detection model used in the current development command is `yolov8n.pt`. Obtain model files only from their official source and review their license and usage restrictions before use.

### 4. Run an attendance session

Example command from the current development setup:

```bash
python attendai_attendance_session_unknown_queue.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --person-model yolov8n.pt \
  --person-imgsz 256 \
  --person-conf 0.40 \
  --face-scale 1.0 \
  --face-every 5 \
  --enable-attendance \
  --session-name "Recognition Test" \
  --attendance-db data/attendai_attendance.db
```

Adjust camera index, resolution, thresholds, and session name for the actual installation. The command is an example from the development workflow; available options may change. Check the script's help output before running it:

```bash
python attendai_attendance_session_unknown_queue.py --help
```

### 5. Run the faculty review dashboard

In a separate terminal:

```bash
cd ~/Desktop/Indiran
source .review_env/bin/activate
python -m streamlit run attendai_faculty_review.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Open `http://127.0.0.1:8501` on the Jetson. To access the dashboard from another device, configure a secure network deployment rather than exposing the development server directly to the public internet.

## Data protection and security

Class Vision may process face images, face embeddings, student identifiers, attendance records, and classroom video. These are sensitive data and must be handled accordingly.

Before any institutional pilot or deployment:

1. Obtain written authorization from the institution and establish an appropriate lawful basis for processing.
2. Give students and relevant stakeholders clear notice about what is collected, why it is collected, who can access it, and how long it is retained.
3. Provide a practical non-biometric/manual attendance alternative.
4. Restrict access using role-based permissions and strong authentication.
5. Encrypt data in transit and at rest; protect keys and credentials outside source control.
6. Set retention and deletion policies for images, embeddings, logs, and attendance records.
7. Maintain audit logs for enrollment, corrections, exports, and administrative actions.
8. Secure cameras, the Jetson device, the local network, backups, and remote access.
9. Validate recognition performance across real classroom conditions and demographic groups. Provide a human correction process for false matches and missed detections.
10. Obtain appropriate legal and institutional review before collecting or processing biometric data, including review of applicable Indian data-protection requirements.

**No biometric data or student records should be uploaded to this public/source repository.** Use synthetic or anonymized test data when sharing examples.

## Model and third-party licensing

Review the current licenses and terms for every model, dataset, framework, and dependency before commercial or institutional use.

In particular:

- The Ultralytics YOLOv8 codebase is distributed under AGPL-3.0, with separate Enterprise licensing available. Assess whether the chosen use and distribution model requires an alternative license or a different detector.
- InsightFace code and pretrained model weights may have different terms. The pretrained `buffalo_l` model pack is described by InsightFace as being for non-commercial research purposes. Obtain appropriate permission or a commercial license before commercial deployment.
- Review the licenses for PyTorch, ONNX Runtime, OpenCV, Streamlit, and all other dependencies.

This README is not legal advice and does not grant rights to any third-party model or software.

## Development notes

- Keep student data, local databases, model weights, generated videos, logs, and environment files out of Git.
- Keep secrets in environment variables or a protected secrets manager; never commit API keys, passwords, tokens, or private SSH keys.
- Do not use unreviewed face matches to automatically make consequential decisions about students.
- Maintain test cases for low light, occlusion, side profiles, multiple people, camera movement, and crowded scenes.
- Measure end-to-end throughput and latency on the actual Jetson device. A CUDA-enabled PyTorch installation does not prove that every model is running on the GPU.
- Before deployment, add documented dependency versions, automated tests, backup/restore procedures, monitoring, and a secure update process.

## Contributing

Contributions should include a clear description of the change and relevant test evidence. Avoid submitting real student data, face images, embeddings, or recordings in issues and pull requests. Use synthetic test fixtures wherever possible.

## License

No project-level license is specified in this repository yet. Unless and until a license is added, the source code is not automatically open for reuse, modification, or redistribution. Third-party components remain subject to their own licenses and terms.

## Maintainer

**Class Vision**  
GitHub: [indhiran08-coder](https://github.com/indhiran08-coder)
