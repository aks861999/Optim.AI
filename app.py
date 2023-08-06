import streamlit as st
import time
import pandas as pd
import numpy as np

import openai

import imaplib
import email
import yaml

import json
import requests
import nltk
nltk.data.path.append('./nltk.txt')
nltk.download('vader_lexicon')
import plotly.express as px



global submitted_login_button, username, df, submitted_keyword_button

imap_url = 'imap.gmail.com'



st.set_page_config(page_title="Optim.AI by Akash",page_icon='📰',layout="wide")



st.write("**Pre-requisites to use this app :-**")
st.markdown("- Get your GMAIL APP Password -----> Login to your Google Account -> Security -> 2 Step Verification -> APP Password")
st.markdown("- Get your OpenAI API Key -----> https://platform.openai.com/signup?launch")





def segment(df):
    ## Form a pandas series with all value counts in the "Label" column in the Dataframe "df" ##
    counts = df.label.value_counts(normalize=True) * 100

    print(counts)
    
    ## Convert pandas series to a dataframe ##
    counts=counts.to_frame()

    print(counts)
    
    ## Form a column named 'Segment' that consist of '+1', '-1' and  '0'  for positive , negative , neutral respectively ##
    counts['segment']=counts.index


    counts.sort_values(by=['segment'],inplace=True)


    #print(counts.label)
    counts['label']=counts.index

    counts.loc[counts['label'] == 0, 'stat'] = 'Neutral'
    counts.loc[counts['label'] == 1, 'stat'] = 'Positive'
    counts.loc[counts['label'] == -1, 'stat'] = 'Negative'

    fig = px.pie(counts, values='proportion', names='stat')
    ## Build the Figure basically a pie chart with graph object of plotly ## 
    #fig = go.Figure(data=[go.Pie(labels=['Negative','Neutral','Positive'], values=counts['label'])])
    #fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    
    ## make two lists for positive and negative news ##
    positive=list(df[df['label'] ==  1].headline)
    negative=list(df[df['label'] == -1].headline)
    
    return (fig,positive,negative)








def sentiment(headlines):
    
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    sia = SIA()
    
    results = []
    
    
    for line in headlines:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)
    
    df = pd.DataFrame.from_records(results)
    
    df['label'] = 0
    df.loc[df['compound'] > 0.17, 'label'] = 1
    df.loc[df['compound'] < -0.17, 'label'] = -1

    #st.dataframe(df)
    
    return(segment(df))








def get_news(news_p):
    payload = {
   'api_key': '1982b65292d6a33120d1e24aab066611',
   'query': news_p,
   'num': '50'
    }
    response = requests.get(
    'https://api.scraperapi.com/structured/twitter/search', params=payload)
    data = response.json()

    df_twits = data['organic_results']

    df_twits = pd.DataFrame.from_records(df_twits)


    
    p=list(df_twits['snippet'])

    print(p)
    
    

    ## Go to the 'sentiment' function for sentiment classification ##
    #select country from country dropdown
    return(sentiment(p),'Germany')



























def get_username(email_address):
	r = email_address.index("@")
	return "".join(l for l in email_address[:r] if l.isalpha())

if st.session_state.get('step') is None:
    st.session_state['step'] = 0
# the number is my 'configuration'
if st.session_state.get('number') is None:
    st.session_state['number'] = 0




st.title('Welcome to Optim.AI')
st.subheader("**- *Unlock the power of data-driven decision making with our inventory management app Optim.AI . Seamlessly extract valuable insights from quotation emails, including delivery dates, brand names, and product prices, empowering you to make the right choices for your B2B business* -**")









with st.sidebar:
    #with st.echo():
    #st.write("This code will be printed to the sidebar.")

    
    with st.form("my_form"):

        #st.write("Inside the form")
        user = st.text_input("write your email ID here")
        password = st.text_input("write your App password", type = "password")
        open_AI_Key = st.text_input("Enter the OpenAI API Key", type = "password")

        openai.api_key = open_AI_Key
        #keyword = st.text_input("write the keyword")

        


        # Every form must have a submit button.

        submitted_login_button = st.form_submit_button("Submit")



if submitted_login_button:
    st.session_state['step'] = 1




if st.session_state['step'] == 1:     
    with st.spinner("Logging in..."):
        time.sleep(0.1)

        # Connection with GMAIL using SSL
        my_mail = imaplib.IMAP4_SSL(imap_url)

        print(my_mail)
        #st.write(my_mail)

        try:
            my_mail.login(user, password)

            username = get_username(user)

            st.write("Hey "+ username +", welcome")

            with st.form('my form1'):
                keyword = st.text_input('Enter the keyword')
                st.session_state['number'] = (my_mail,keyword)
            
                if st.form_submit_button("Get Result"):            
                    st.session_state['step'] = 2
                    st.experimental_rerun()  # form should not be shown after clicking 'save' 
        
        except Exception as e:
            exep=str(e)
            print(exep)
            st.write(exep)

        

if st.session_state['step'] == 2:
    with st.spinner("Let the Optim Fetch the Details.."):
        time.sleep(5)





    my_mail = st.session_state["number"][0]
    keyword = st.session_state["number"][1]
    
    print(keyword)
    #st.write(keyword)
    keyword = "'"+keyword+"'"
    my_mail.select('Inbox')
    _, data = my_mail.search(None, 'SUBJECT', keyword)  #Search for emails with specific key and value

    
    mail_id_list = data[0].split()  #IDs of all emails that we want to fetch

    msgs = [] # empty list to capture all messages

    for num in mail_id_list:
        typ, data = my_mail.fetch(num, '(RFC822)') #RFC822 returns whole message (BODY fetches just body)
        msgs.append(data)
    
    #st.write('msgs are below')
    print(msgs)
    #st.write(msgs)


    df = pd.DataFrame(columns=['Sender','Subject','Body of the email','Delivery date','Quoted price','Brand'])

    mail_count = 0

    for msg in msgs[::-1]:

        for response_part in msg:

            if type(response_part) is tuple:



                extracted = []



                my_msg=email.message_from_bytes((response_part[1]))
                #st.write("_________________________________________")
                #st.write ("subj:", my_msg['subject'])
                extracted.append(my_msg['from'])
                extracted.append(my_msg['subject'])
                #st.write ("from:", my_msg['from'])
                #st.write ("body:")
                for part in my_msg.walk():
                    #print(part.get_content_type())
                    if part.get_content_type() == 'text/plain':
                        body_part=part.get_payload()
                        body_part=body_part.replace('\r\n', ' ')
                        body_part=body_part.replace('=E2=82=AC', '')
                        body_part=body_part.replace('$', '')
                        

                #st.write(body_part)
                extracted.append(body_part)

                #openai

                prompt = "Extract Delivery date in YYYY-MM-DD format, quoted price, brand from the email text:\n\n    " + body_part

                response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.5,
                max_tokens=50,
                top_p=1.0,
                frequency_penalty=0.8,
                presence_penalty=0.0
                )

                response = response["choices"][0]['text']

                modified_response = response

                modified_response = modified_response.split('\n')
                modified_response=modified_response[2:]


                for i in modified_response:
                    i = i.split(':')
                    extracted.append(i[1])
                    print(i[1])

        df.loc[mail_count] = extracted
        mail_count += 1
	if mail_count == 3:
		time.sleep(60)
		print(waiting for_1 minute)

    #st.dataframe(df)    
    df['Quoted price']=df['Quoted price'].replace(r',',r'', regex=True)
    df['Delivery date'] =  pd.to_datetime(df['Delivery date'] , format='%Y%m%d', errors='ignore')
    df['Quoted price'] =  pd.to_numeric(df['Quoted price'])                    

    st.header("Overall Emails")

    st.dataframe(df)

    st.write(" ")
    st.write(" ")
    

    st.header("Earliest possible Option")

    df_date=df.sort_values(by = 'Delivery date')

    row_df_date = df_date.iloc[0]

    row_df_date_dict = row_df_date.to_dict()

    st.text("Sender Name : - "+ row_df_date_dict['Sender'])

    st.text("Subject of the email : - "+ row_df_date_dict['Subject'])

    st.text("Product Brand "+ row_df_date_dict['Brand'])

    st.write("**Expected Delivery Date: - "+ str(row_df_date_dict['Delivery date']) + "!**")

    st.text("Price offered : - "+ str(row_df_date_dict['Quoted price'])) 
    
    st.text("Email Content : - "+ row_df_date_dict['Body of the email']) 
    
    st.write(" ")
    st.write(" ")

    st.header("Cheapest Option")

    df_price=df.sort_values(by = 'Quoted price')

    row_df_price = df_price.iloc[0]

    row_df_price_dict = row_df_price.to_dict()

    st.text("Sender Name : - "+ row_df_price_dict['Sender'])

    st.text("Subject of the email : - "+ row_df_price_dict['Subject'])

    st.text("Product Brand "+ row_df_price_dict['Brand'])

    st.write("**Price offered : - "+ str(row_df_price_dict['Quoted price']) + "!**")

    st.text("Expected Delivery Date: - "+ str(row_df_price_dict['Delivery date']))

    st.text("Email Content : - "+ row_df_price_dict['Body of the email'])

    comapny_list = df['Brand']


    st.write("")
    st.write("")
    st.header('Would you like to witness brand reviews ?')
    input_from_user = st.selectbox(
    'Select the brand',
    comapny_list)










    ##########################################

    x=get_news(input_from_user)
    print(x)
    ## the object 'x' gets list consist of a tuple (figure,positive news,negative news) and the country name ##
    fig=x[0][0]
    pos=x[0][1]
    neg=x[0][2]
    country_name=x[1]

    st.header('The Positive Reviews about ' + input_from_user +' are')



    for i in pos[:5]:
        st.text(i)

    st.header('Pie Chart Visualisation')
        
    st.plotly_chart(fig)

    st.header('The Negative Reviews about ' + input_from_user +' are')

    for i in neg[:5]:
        st.text(i)
