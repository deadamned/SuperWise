import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/plagiarism/')
def plagiarism():
    return render_template('plagiarism.html')

@app.route('/login', methods=['POST'])
def login():
   text2 = request.form['text2']
   cat = request.form['cat']
   if cat == "tadeusz2":
      url = "https://sciaga.pl/tekst/104389-105-pan-tadeusz-jako-epopeja-narodowa"
      response = requests.get(url)
      html = response.content 
      soup = BeautifulSoup(html)
      tadeusz = soup.find('p', attrs={'itemprop' : 'articleBody'})
      with open('file.txt', 'w') as f:
         f.write(tadeusz.text.replace('<br>', ''))

      f1 = open('file.txt')
      f2 = open(text2)

      f1_line = f1.readline()
      f2_line = f2.readline()

      line_no = 1

      while f1_line != '' or f2_line != '':
         f1_line = f1_line.rstrip()
         f2_line = f2_line.rstrip()
         if f1_line != f2_line:
            if f2_line !='':
               with open('file3.txt', 'a') as f:
                  f.write(f2_line)
                  f.write("\n")  
         f1_line = f1.readline()
         f2_line = f2.readline()
         line_no += 1
      f1.close()
      f2.close()
      with open('file3.txt', 'r') as myfile:
         data = myfile.read()
      
      if data != "":
         return data
      else:
         url = "https://sciaga.pl/tekst/61302-62-pan_tadeusz_adama_mickiewicza_jako_epopeja_narodowa"
         response = requests.get(url)
         html = response.content 
         soup = BeautifulSoup(html)
         tadeusz = soup.find('p', attrs={'itemprop' : 'articleBody'})
         with open('file.txt', 'w') as f:
            f.write(tadeusz.text.replace('<br>', ''))

         f1 = open('file.txt')
         f2 = open(text2)

         f1_line = f1.readline()
         f2_line = f2.readline()

         line_no = 1

         while f1_line != '' or f2_line != '':
            f1_line = f1_line.rstrip()
            f2_line = f2_line.rstrip()
            if f1_line != f2_line:
               if f2_line !='':
                  with open('file3.txt', 'a') as f:
                     f.write(f2_line)
                     f.write("\n")  
            f1_line = f1.readline()
            f2_line = f2.readline()
            line_no += 1
         f1.close()
         f2.close()
         with open('file3.txt', 'r') as myfile:
            data = myfile.read()
         
         if data != "":
            return data
         else:
            url = "https://sciaga.pl/tekst/41443-42-pan_tadeusz_epopeja_narodowa"
            response = requests.get(url)
            html = response.content 
            soup = BeautifulSoup(html)
            tadeusz = soup.find('p', attrs={'itemprop' : 'articleBody'})
            with open('file.txt', 'w') as f:
               f.write(tadeusz.text.replace('<br>', ''))

            f1 = open('file.txt')
            f2 = open(text2)

            f1_line = f1.readline()
            f2_line = f2.readline()

            line_no = 1

            while f1_line != '' or f2_line != '':
               f1_line = f1_line.rstrip()
               f2_line = f2_line.rstrip()
               if f1_line != f2_line:
                  if f2_line !='':
                     with open('file3.txt', 'a') as f:
                        f.write(f2_line)
                        f.write("\n")  
               f1_line = f1.readline()
               f2_line = f2.readline()
               line_no += 1
            f1.close()
            f2.close()
            with open('file3.txt', 'r') as myfile:
               data = myfile.read()
         
            if data != "":
               return data

@app.route('/correct/')
def correct():
    return render_template('correct.html')


   

if __name__ == '__main__':
    app.run(debug=True)