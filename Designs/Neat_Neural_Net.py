class NEATNEURALNET:
    class Subclassone:        
        # XOR inputs and expected output values
        XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        XOR_OUTPUTS = [(0.0,), (1.0,), (1.0,), (0.0,)]
        
        @staticmethod
        def eval_fitness(net):
            """
            Evaluate fitness of a neural network on XOR problem.
            Returns higher fitness for better performance.
            """
            error_sum = 0.0
            
            for xi, xo in zip(NEATNEURALNET.Subclassone.XOR_INPUTS, 
                             NEATNEURALNET.Subclassone.XOR_OUTPUTS):
                try:
                    output = net.activate(xi)
                    # Handle single output or multiple outputs
                    predicted = output[0] if isinstance(output, (list, tuple)) else output
                    expected = xo[0]
                    error_sum += abs(predicted - expected)
                except Exception as e:
                    print(f"Error during network activation: {e}")
                    return 0.0
            
            # Calculate amplified fitness (max fitness = 16 when error = 0)
            fitness = max(0.0, (4.0 - error_sum) ** 2)
            return fitness
        
        @staticmethod
        def test_network(net, verbose=False):
            """Test network on all XOR cases and return accuracy."""
            correct = 0
            total = len(NEATNEURALNET.Subclassone.XOR_INPUTS)
            
            for i, (xi, xo) in enumerate(zip(NEATNEURALNET.Subclassone.XOR_INPUTS,
                                           NEATNEURALNET.Subclassone.XOR_OUTPUTS)):
                output = net.activate(xi)
                predicted = output[0] if isinstance(output, (list, tuple)) else output
                expected = xo[0]
                
                # Consider prediction correct if within threshold
                is_correct = abs(predicted - expected) < 0.5
                if is_correct:
                    correct += 1
                
                if verbose:
                    print(f"Input: {xi}, Expected: {expected}, Got: {predicted:.3f}, "
                          f"Correct: {is_correct}")
            
            accuracy = correct / total
            return accuracy