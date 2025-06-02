
#.py client
import grpc
import echo_pb2
import echo_pb2_grpc

def generate_messages():
    """
    生成流式请求数据
    """
    messages = ["Hello", "World", "This", "is", "a", "bidirectional", "stream"]
    for msg in messages:
        yield echo_pb2.EchoRequest(message=msg)

def run():
    # 连接到服务端
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = echo_pb2_grpc.EchoServiceStub(channel)
        # 调用双向流式接口
        responses = stub.BidirectionalStream(generate_messages())
        for response in responses:
            print(f"Received response: {response.message}")

if __name__ == "__main__":
    run()