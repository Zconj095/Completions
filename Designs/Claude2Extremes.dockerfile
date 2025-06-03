# Dockerfile
FROM python:3.8  
COPY . /app  
RUN pip install -r requirements.txt

EXPOSE 5000  

# Build container
docker build -t analytics .  

# Kubernetes pod 
apiVersion: v1
kind: Pod
metadata:
  name: analytics
spec:
  containers:
    - name: analytics
      image: analytics

npm install react-native

# Components 
import {View, Text} from 'react-native'

export default App = () => {

  return (
    <View> 
      <Text>My App</Text>
    </View>
  )
}