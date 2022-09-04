# from unicodedata import name
# import mysql.connector

#host="localhost",
#user="root", or HuuNhat
#password="0302"
# mydb = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="0302"
# )
# mycursor = mydb.cursor()
# #mycursor.execute("DROP DATABASE mydatabase")
# print(mycursor)
import mysql.connector
from mysql.connector import errorcode
config = {
  'user': 'root',
  'password': '0302',
  'host': 'localhost',
  #'database': 'employees',
  'raise_on_warnings': True
}
try:
  cnx = mysql.connector.connect(**config)
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cnx.close()
print(cnx)