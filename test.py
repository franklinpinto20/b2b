import streamlit as st
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import auth
from datetime import date


cred = credentials.Certificate("pondering-5ff7c-c033cfade319.json")
firebase_admin.initialize_app(cred)
def app():
# Usernm = []
    st.title('Welcome to :violet[B2B] ::')

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''



    def f(): 
        try:
            user = auth.get_user_by_email(email)
            print(user.uid)
            st.session_state.username = user.uid
            st.session_state.useremail = user.email
            st.session_state.displayName = user.display_name
            
      
            
            global Usernm
            Usernm=(user.uid)
            
            st.session_state.signedout = True
            st.session_state.signout = True    
  
            
        except: 
            st.warning('Login Failed')

    def t():
        st.session_state.signout = False
        st.session_state.signedout = False   
        st.session_state.username = ''


        
    
        
    if "signedout"  not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False    
        

        
    
    if  not st.session_state["signedout"]: # only show if the state is False, hence the button has never been clicked
        choice = st.selectbox('Login/Signup',['Login','Sign up'])
        email = st.text_input('Email Address')
        password = st.text_input('Password',type='password')
        

        
        if choice == 'Sign up':
            username = st.text_input("Enter  your unique username")
            gender=st.selectbox('Enter gender',['m','f'])
            age=st.text_input('Enter age ')
            country=st.selectbox('Enter country ',['Afghanistan','Albania','Germany','Andorra','Angola','Antigua and Barbuda','Saudi Arabia','Algeria','Argentina','Armenia','Australia','Austria' ,'Azerbaijan','Bahamas','Bangladesh','Barbados','Bahrain','Belgium','Belize','Benin','Belarus','Burma','Bolivia','Bosnia and Herzegovina' ,'Botswana','Brazil','Brunei','Bulgaria','Burkina Faso','Burundi','Bhutan','Cape Verde','Cambodia','Cameroon','Canada','Qatar' ,'Chad','Chile','China','Cyprus','Vatican City','Colombia','Comoros','North Korea','South Korea','Ivory Coast',' Costa Rica','Croatia','Cuba','Denmark','Dominica','Ecuador','Egypt','El Salvador','United Arab Emirates','Eritrea','Slovakia','Slovenia' ,'Spain','United States','Estonia','Ethiopia','Philippines','Finland','Fiji','France','Gabon','Gambia','Georgia','Ghana', 'Grenada','Greece','Guatemala','Guyana','Guinea','Equatorial Guinea','Guinea-Bissau','Haiti','Honduras','Hungary','India','Indonesia' ,'Iraq','Iran','Ireland','Iceland','Marshall Islands','Solomon Islands','Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan' ,'Kenya','Kyrgyzstan','Kiribati','Kuwait','Laos','Lesotho','Latvia','Lebanon','Liberia','Libya','Liechtenstein','Lithuania',' Luxembourg','North Macedonia','Madagascar','Malaysia','Malawi','Maldives','Maldives','Malta','Morocco','Mauritius','Mauritania','Mexico',' Micronesia','Moldova','Monaco','Mongolia','Montenegro','Mozambique','Namibia','Nauru','Nepal','Nicaragua','Niger','Nigeria','Norway' ,'New Zealand','Oman','Netherlands','Pakistan','Palau','Panama','Papua New Guinea','Paraguay','Peru','Poland','Portugal',' United Kingdom','Central African Republic','Czech Republic','Republic of the Congo','Democratic Republic of the Congo','Dominican Republic','South African Republic','Rwanda','Romania','Russia',' Samoa','Saint Kitts and Nevis','San Marino','Saint Vincent and the Grenadines','Saint Lucia','Sao Tome and Principe','Senegal','Serbia','Seychelles','Sierra Leone' ,'Singapore','Syria','Somalia','Sri Lanka','Swaziland','Sudan','South Sudan','Sweden','Switzerland','Suriname','Thailand',' Tanzania','Tajikistan','East Timor','Togo','Tonga','Trinidad and Tobago','Tunisia','Turkmenistan','Turkey','Tuvalu','Ukraine','Uganda', 'Uruguay','Uzbekistan','Vanuatu','Venezuela','Vietnam','Yemen','Djibouti','Zambia','Zimbabwe'])
            musicType=st.selectbox('Enter your type of music',['Metal','Rock','Reggae'])
            today=today = date.today()
            d2 = today.strftime("%B %d, %Y")
            displayName=gender+"-"+age+"-"+country+"-"+d2+"-"+musicType
            if st.button('Create my account'):
                user = auth.create_user(email = email, password = password,uid=username,display_name=displayName)
                
                st.success('Account created successfully!')
                st.markdown('Please Login using your email and password')
                st.balloons()
        else:
            # st.button('Login', on_click=f)          
            st.button('Login', on_click=f)
            
            
    if st.session_state.signout:
                st.text('Name '+st.session_state.username)
                st.text('Email id: '+st.session_state.useremail)
                st.text('Detalles: '+st.session_state.displayName)

                addInfo=st.session_state.displayName
                #st.text('Detalles: '+addInfo)

                resto=addInfo
                gender=resto.split("-",1)[0]
                resto=resto.split("-",1)[1]
                age=resto.split("-",1)[0]
                resto=resto.split("-",1)[1]
                country=resto.split("-",1)[0]
                resto=resto.split("-",1)[1]
                fdate=resto.split("-",1)[0]
                resto=resto.split("-",1)[1]
                musicType=resto.split("-",1)[0]
            
                st.text('gender: '+gender)
                st.text('age: '+age)
                st.text('country: '+country)
                st.text('Date: '+fdate)
                st.text('Music Type: '+musicType)

                st.button('Sign out', on_click=t)           
                   

                            
    def ap():
        st.write('Posts')
