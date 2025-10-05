from flask import Flask , render_template , request  , send_from_directory , url_for
import os
from faceBlur import faceBlurrer

app = Flask(__name__)

@app.route('/' , methods=['GET','POST'])
def index():
    download_link = None
    if request.method == 'POST':
        file = request.files['video']
        filepath = os.path.join('uploads',file.filename)
        file.save(filepath)

        kernal = int(request.form.get('kernal',35))
        confidence = float(request.form.get('confidence',0.7))

        output_filename = "blurred_" + file.filename
        output_path = os.path.join('outputs',output_filename)
        faceBlurrer(filepath,output_path,confidence,kernal)

        download_link = url_for('download_file',filename = output_filename)

    return render_template('index.html', download_link=download_link)

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory('outputs', filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)