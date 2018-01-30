abstract class Layer {
    private Vector activation;

    Layer(int inputs, int outputs) {
        activation = new Vector(outputs);
    }

    abstract void activate(Vector weights, Vector x);

    protected void setActivation(Vector vector) {
      this.activation = vector;
    }

    public Vector getActivation() {
        return activation;
    }
}
