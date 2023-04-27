import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.math.Add;

import java.nio.charset.StandardCharsets;

public class TensorFlowExample {
    public static void main(String[] args) throws Exception {
        // Construct the computation graph
        try (Graph graph = new Graph()) {
            // Create a constant tensor with a string value
            String message = "Hello, TensorFlow!";
            byte[] messageBytes = message.getBytes(StandardCharsets.UTF_8);
            long[] shape = new long[] {messageBytes.length};
            Tensor<String> inputTensor = (Tensor<String>) Tensor.create(String.class);
            Constant<String> constant = Ops.create(graph).constant(inputTensor.bytesValue());

            // Add a constant tensor to itself
            Add<String> add = Ops.create(graph).math.add(constant.asOutput(), constant.asOutput());

            // Run the computation graph in a session
            try (Session session = new Session(graph)) {
                Tensor<String> outputTensor = (Tensor<String>) session.runner().fetch(add.asOutput()).run().get(0);
                byte[] outputBytes = new byte[(int) outputTensor.shape()[0]];
                outputTensor.copyTo(outputBytes);
                String outputMessage = new String(outputBytes, StandardCharsets.UTF_8);
                System.out.println(outputMessage);
            }
        }
    }
}
