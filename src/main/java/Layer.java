abstract class Layer {
    protected Vector activation;

    Layer(int inputs, int outputs) {
        activation = new Vector(outputs);
    }

    abstract void activate(Vector weights, Vector x);
}
