
# Serverless Video Cleaning Tool (S3 -> Lambda -> S3)

## Introduction

This project provides a tool to automatically clean video files by removing segments with low motion activity. It includes:

1.  A Python script (`test.py`) for local execution.
2.  An AWS Lambda function (`lambda_function.py`) triggered by S3 uploads, which performs the cleaning in a serverless environment and stores the result in another S3 bucket.

The core cleaning logic uses OpenCV for motion analysis and FFmpeg for video segment extraction and concatenation.

## Features

* Detects motion based on frame differences.
* Smooths motion scores over time to reduce noise sensitivity.
* Identifies contiguous segments above a specified activity threshold and minimum duration.
* Applies a buffer time around active segments (optional).
* Uses FFmpeg for efficient, lossless segment extraction (where possible) and concatenation.
* Serverless deployment option using AWS Lambda and S3 triggers.

## Architecture Options

1.  **Local Execution:** Run `test.py` directly on your machine, providing input/output file paths and parameters via command-line arguments. Requires local installation of Python libraries and FFmpeg.
2.  **AWS Lambda:** Upload videos to a source S3 bucket. An S3 trigger invokes the Lambda function, which uses an FFmpeg Lambda Layer, processes the video, and uploads the result to a destination S3 bucket.

## Prerequisites

* Git (for cloning the repository)
* Python 3.8+
* pip (Python package installer)
* **For AWS Lambda Setup:**
    * An AWS Account
    * AWS CLI installed and configured (`aws configure`)

## Setup Instructions

Choose the setup method based on how you intend to run the project:

### 1. Local Setup (Running `test.py` on your machine)

Follow these steps to run the video cleaning script directly on your computer.

**a) Clone the Repository:**

```bash
git clone <your-repository-url> # Replace with your repo URL
cd <repository-directory>
```
b) Create a Python Virtual Environment (Recommended):

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows (cmd/powershell)
python -m venv venv
.\venv\Scripts\activate
```
c) Install Python Libraries:

Create a requirements.txt file in your project directory with the following content:
```bash
# requirements.txt
numpy
opencv-python-headless
ffmpeg
```
Note: opencv-python-headless is used as it doesn't require GUI libraries and is suitable for server/scripting tasks. If you needed OpenCV GUI functions like cv2.imshow, you would install opencv-python instead.

d) Install FFmpeg:

The script calls the ffmpeg command directly. You need to install FFmpeg and ensure it's available in your system's PATH.

Check Installation: Open your terminal/command prompt and run:
```bash
ffmpeg -version
```
If it shows version info, you're likely set. If not, install it below.

Installation:

Linux (Debian/Ubuntu):
```bash
sudo apt update && sudo apt install ffmpeg
```
macOS (using Homebrew): If you don't have Homebrew, install it from https://brew.sh/.
```bash
brew install ffmpeg
```
Windows:
Download a static build from gyan.dev or the official FFmpeg site.
Extract the archive (e.g., to C:\ffmpeg).
Add the bin directory inside the extracted folder (e.g., C:\ffmpeg\bin) to your system's PATH environment variable. (Search Windows for "Edit the system environment variables").
Restart your terminal/command prompt after adding to PATH.
Verify Installation: Run ffmpeg -version again in a new terminal window.

e) Run the Script: (See Usage section below)
__________
2. AWS Lambda Setup (Serverless Deployment)

Follow these steps to deploy the video cleaning logic as an AWS Lambda function.

a) Prerequisites:

AWS Account
AWS CLI installed and configured (aws configure)
b) Python Libraries (Packaging):

Lambda needs the numpy and opencv-python-headless libraries packaged with your function code. boto3 is provided by the Lambda runtime.

Create a project directory (e.g., lambda_video_cleaner).
Copy your lambda_function.py file into this directory.
Create requirements.txt (same as local setup) inside the directory.
Install packages into the directory:
```bash
cd lambda_video_cleaner
pip install -r requirements.txt -t .
```
5. Create the deployment ZIP (ensure lambda_function.py is at the root):
```bash
zip -r ../lambda_deployment_package.zip .
```
c) FFmpeg Lambda Layer:

Lambda doesn't include FFmpeg. You need to provide it via a Layer.

Obtain FFmpeg Static Binary: Get an ffmpeg executable compatible with Amazon Linux 2 (x86_64). See [Episode 2](<link-to-your-lab-guide-episode-2> or describe here) of the detailed Lab Guide (or search John Van Sickle FFmpeg builds, BtbN builds).
Create Layer Structure:
```bash
ffmpeg_layer/
└── bin/
    └── ffmpeg  # Place the static binary here
```
3. Set Execute Permissions (Linux/macOS): chmod +x ffmpeg_layer/bin/ffmpeg
4. Zip Layer Contents:
```bash
cd ffmpeg_layer
zip -r ../ffmpeg_layer.zip .
cd ..
```
5. Publish Layer
```bash
aws lambda publish-layer-version \
    --layer-name ffmpeg-static \
    --description "Static FFmpeg binary for Lambda" \
    --zip-file fileb://ffmpeg_layer.zip \
    --compatible-runtimes python3.12 python3.11 python3.10 python3.9 python3.8 \
    --region YOUR_AWS_REGION
```
d) Deploy Lambda Function & Resources:

This involves creating S3 buckets, an IAM execution role, the Lambda function itself (uploading the code ZIP, attaching the FFmpeg layer, setting environment variables like DESTINATION_BUCKET, increasing memory/timeout), and configuring the S3 trigger.
   





