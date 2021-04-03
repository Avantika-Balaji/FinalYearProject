f=open("captions.txt","r")
a=f.read()
cap=a.split("==*")


for caps in cap:
	temp=caps.split('=')
	print(temp)