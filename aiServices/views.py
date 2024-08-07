from django.shortcuts import render
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .utils import ReplicateClient,StableDiffusionClient,HuggingFaceClient,ClaudeClient,ZeroGPTClient

class Home(APIView):
    def get(self,request):
        return render(request,template_name='index.html')
    
class PromptInteractionAPIView(APIView):
    
    def post(self,request,*args, **kwargs):
        prompt = request.data.get('prompt',None)
        service_type = request.data.get('service_type',None)
        
        if prompt and service_type :
            if service_type == 'replicate':
                client = ReplicateClient(settings.REPLICATE_API_KEY)
                output = client.generate_image(prompt)
                response= {
                    "message" : "Success",
                    "data" : output
                }
                return Response(response,status=status.HTTP_200_OK)
            elif service_type == 'stable_diffusion':
                
                client = StableDiffusionClient(settings.STABLE_DIFFUSION_API_KEY)
                output = client.generate_image(prompt)
                if output : 
                    response = {
                        "message" : "Success",
                        "data" : output
                    }
                    return Response(response,status=status.HTTP_200_OK)
                return Response({"Error" : "Bad Request"},status=status.HTTP_400_BAD_REQUEST)
            
            elif service_type == 'hugging_face':
                print("Hugging Face Client Working")
                client = HuggingFaceClient(settings.HUGGING_FACE_API_KEY)
                output = client.generate_text(prompt)
                if output : 
                    response = {
                        "message" : "Success",
                        "data" : output
                    }
                    return Response(response,status=status.HTTP_200_OK)
                return Response({"Error" : "Bad Request"},status=status.HTTP_400_BAD_REQUEST)
            elif service_type == 'chatbot':
                client = ClaudeClient(settings.CLAUDE_API_KEY)
                output = client.generate_response(prompt)
                if output : 
                    response = {
                        "message" : "Success",
                        "data" : output
                    }
                    return Response(response,status=status.HTTP_200_OK)
                return Response({"Error" : "Bad Request"},status=status.HTTP_400_BAD_REQUEST)
            else:
                #zeroGPT
                client = ZeroGPTClient(settings.ZERO_GPT_API_KEY)
                output = client.detect_ai_content(text=prompt)
                if output : 
                    response = {
                        "message" : "Success",
                        "data" : output
                    }
                    return Response(response,status=status.HTTP_200_OK)
                return Response({"Error" : "Bad Request"},status=status.HTTP_400_BAD_REQUEST)
                
        response = {
            "message" : "Failure",
            "data" : "Bad Request"
        }
        return Response(response,status=status.HTTP_400_BAD_REQUEST)