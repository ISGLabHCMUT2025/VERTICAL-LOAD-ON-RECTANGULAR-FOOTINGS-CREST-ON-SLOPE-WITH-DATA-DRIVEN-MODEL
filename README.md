# VERTICAL-LOAD-ON-RECTANGULAR-FOOTINGS-CREST-ON-SLOPE-WITH-DATA-DRIVEN-MODEL

Includes a GUI for entering rectangular footings on Hoek-Brown slope parameters to predict load capacity

First, it should be clarified that the GUI runs through the Anaconda Prompt, therefore you need to install Anaconda beforehand. Or you can also use any other IDLE or can use CMD directly of window. And all the files need to be downloaded.

Second, you need to download the file thebestmodel.pkl to establish a "calculation factory." Once this factory is set up, any input data will be processed through it and return the corresponding prediction results.

Third, open the Anaconda Prompt and navigate or CMD with adminisrtator to the directory that contains the App.py file.

Finally, enter the following command in the prompt: "python -m streamlit run app.py --server.port 8501"

Once the tool is launched, you can easily input the parameters of your choice and click the Predict button to obtain results.

  Notes:
  
  *** Make sure you have installed all the required libraries used in the App.py file. You can do this by opening App.py with Notepad, checking which libraries are imported, and installing them accordingly. You may also install the required libraries using the Library_essential.txt file. Simply copy each library listed in the file and paste it into the Anaconda Prompt to install them one by one.
  
  *** Sometimes server.port 8501 may not work. Since this is the default port of the Streamlit library, please refer to the Streamlit documentation if you encounter error.

  *** Keep in mind that the predictive model achieves an accuracy level of R² = 99.93%, which means the results should be interpreted as predictions.


Development Team:

Assoc. Prof.:   Van Qui Lai, Vietnam national university Ho Chi Minh city - Ho Chi Minh university of technology

Assoc. Prof.:   Suraparb Keawsawasvong, Thammasat School of Engineering, Thammasat University

Dr. Sc.:        Trong Nghia Le, Ho Chi Minh City University of Transport

Dr. Sc.:        Duy Tan Tran, Ho Chi Minh City University of Transport

Msc. Student:   Huu Nghia Bui, Vietnam national university Ho Chi Minh city - Ho Chi Minh university of technology

Msc. Student:   Hoang Nghi Le, Vietnam national university Ho Chi Minh city - Ho Chi Minh university of technology
