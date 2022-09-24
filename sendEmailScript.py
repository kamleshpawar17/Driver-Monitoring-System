import argparse
import glob
import os
import shutil
import smtplib
import time
from email import encoders
from email.mime.multipart import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate


def sendEmail(login, password, send_to, send_from, subject, text, send_file, server='smtp.gmail.com', port=587):
    # https://www.google.com/settings/security/lesssecureapps
    # accounts.google.com/DisplayUnlockCaptcha
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(send_file, "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(send_file))
    msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    smtp.set_debuglevel(1)
    smtp.ehlo()
    smtp.starttls()
    smtp.login(login, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--email", type=str, default='kamlesh.170184@gmail.com')
    fromemail = 'Drowsiness Detection Event'
    loginname = 'sigularity.ai@gmail.com'
    loginpassword = 'bio17medical'
    toemail = args['email']
    text = 'Dear Customer,' \
           '\n\n\tAttached is a video instance of a drowsiness event captured by the DSMS system.' \
           '\n\n--\nRegards,\nCustomer Support \nSingularity AI'
    nMax2Keep = 500
    while True:
        fnames2send = sorted(glob.glob('./savedVideo/unsend/*.mp4'))
        time.sleep(5.0)
        for f in fnames2send:
            try:
                sendEmail(login=loginname, password=loginpassword, send_to=toemail, send_from=loginname, subject=fromemail,
                     text=text, send_file=f)
                shutil.move(f, './savedVideo/send/')
            except:
                pass
        fnamessend = sorted(glob.glob('./savedVideo/send/*.mp4'))
        if len(fnamessend) > nMax2Keep:
            for f in fnamessend[nMax2Keep:]:
                os.remove(f)
        fnamesunsend = sorted(glob.glob('./savedVideo/unsend/*.mp4'))
        if len(fnamesunsend) > nMax2Keep:
            for f in fnamessend[nMax2Keep:]:
                os.remove(f)

