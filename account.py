import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from datetime import date

cred=credentials.Certificate('taller1-recsys-1efce76b90cd.json')
#firebase_admin.initialize_app(cred)

def app():
    st.title('Welcome  to :blue[B2B] ')

    choice=st.selectbox('Login/Signup',['Login','Sign Up'])

    def f():
        try:
            user=auth.get_user_by_email(email)
            #print (user.iud)
            st.write('Login Succesfull')
        except:
            st.warning('Login Failed') 


    if choice =='Login':
        email=st.text_input('Email Address')
        password=st.text_input('Password',type='password')

        st.button('Login', on_click=f)

    else:

        email=st.text_input('Email Address')
        password=st.text_input('Password',type='password')
        username=st.text_input('Enter your unique username')
        gender=st.selectbox('Enter gender',['m','f'])
        age=st.text_input('Enter age ')
        country=st.selectbox('Enter country ',['Afganistán','Albania','Alemania','Andorra','Angola','Antigua y Barbuda','Arabia Saudita','Argelia','Argentina','Armenia','Australia','Austria','Azerbaiyán','Bahamas','Bangladés','Barbados','Baréin','Bélgica','Belice','Benín','Bielorrusia','Birmania','Bolivia','Bosnia y Herzegovina','Botsuana','Brasil','Brunéi','Bulgaria','Burkina Faso','Burundi','Bután','Cabo Verde','Camboya','Camerún','Canadá','Catar','Chad','Chile','China','Chipre','Ciudad del Vaticano','Colombia','Comoras','Corea del Norte','Corea del Sur','Costa de Marfil','Costa Rica','Croacia','Cuba','Dinamarca','Dominica','Ecuador','Egipto','El Salvador','Emiratos Árabes Unidos','Eritrea','Eslovaquia','Eslovenia','España','Estados Unidos','Estonia','Etiopía','Filipinas','Finlandia','Fiyi','Francia','Gabón','Gambia','Georgia','Ghana','Granada','Grecia','Guatemala','Guyana','Guinea','Guinea ecuatorial','Guinea-Bisáu','Haití','Honduras','Hungría','India','Indonesia','Irak','Irán','Irlanda','Islandia','Islas Marshall','Islas Salomón','Israel','Italia','Jamaica','Japón','Jordania','Kazajistán','Kenia','Kirguistán','Kiribati','Kuwait','Laos','Lesoto','Letonia','Líbano','Liberia','Libia','Liechtenstein','Lituania','Luxemburgo','Macedonia del Norte','Madagascar','Malasia','Malaui','Maldivas','Maldivas','Malta','Marruecos','Mauricio','Mauritania','México','Micronesia','Moldavia','Mónaco','Mongolia','Montenegro','Mozambique','Namibia','Nauru','Nepal','Nicaragua','Níger','Nigeria','Noruega','Nueva Zelanda','Omán','Países Bajos','Pakistán','Palaos','Panamá','Papúa Nueva Guinea','Paraguay','Perú','Polonia','Portugal','Reino Unido','República Centroafricana','República Checa','República del Congo','República Democrática del Congo','República Dominicana','República Sudafricana','Ruanda','Rumanía','Rusia','Samoa','San Cristóbal y Nieves','San Marino','San Vicente y las Granadinas','Santa Lucía','Santo Tomé y Príncipe','Senegal','Serbia','Seychelles','Sierra Leona','Singapur','Siria','Somalia','Sri Lanka','Suazilandia','Sudán','Sudán del Sur','Suecia','Suiza','Surinam','Tailandia','Tanzania','Tayikistán','Timor Oriental','Togo','Tonga','Trinidad y Tobago','Túnez','Turkmenistán','Turquía','Tuvalu','Ucrania','Uganda','Uruguay','Uzbekistán','Vanuatu','Venezuela','Vietnam','Yemen','Yibuti','Zambia','Zimbabue'])
        musicType=st.selectbox('Enter your type of music',['Metal','Rock','Reggae'])
        today=today = date.today()
        d2 = today.strftime("%B %d, %Y")
        displayName=gender+"-"+age+"-"+country+"-"+d2+"-"+musicType
        if st.button('Create my account'):
            user=auth.create_user(emil=email,password=password, uid=username,display_name=displayName)

            st.success('Account created successfully')
            st.markdown('Please Login using your email and password')
            st.balloons()
