from rest_framework.views import APIView
from rest_framework.response import Response

class CBIRView(APIView):
    def post(self, request):
        return Response({'menesaje': 'Holaaaaa'})
