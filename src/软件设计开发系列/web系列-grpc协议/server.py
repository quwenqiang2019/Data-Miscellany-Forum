# server.py
from concurrent import futures
import grpc
import echo_pb2
import echo_pb2_grpc

class EchoService(echo_pb2_grpc.EchoServiceServicer):
    def BidirectionalStream(self, request_iterator, context):
        """
        处理双向流式请求
        """
        for request in request_iterator:
            print(f"Received message: {request.message}")
            # 将客户端发送的消息原样返回
            yield echo_pb2.EchoResponse(message=request.message)

def serve():
    # 创建gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    echo_pb2_grpc.add_EchoServiceServicer_to_server(EchoService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Echo Service started on port 50051.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()