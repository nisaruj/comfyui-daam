// src/reshape.ts
function reshape(flat, shape, fortranOrder = false) {
  if (!shape.length) return flat[0];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  if (flat.length !== totalSize) {
    throw new Error(`Cannot reshape array of size ${flat.length} into shape (${shape.join(", ")})`);
  }
  const strides = new Array(shape.length);
  let stride = 1;
  if (fortranOrder) {
    for (let i = 0; i < shape.length; i++) {
      strides[i] = stride;
      stride *= shape[i];
    }
  } else {
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
  }
  function build(dims, currentCoords) {
    const [currentDim, ...remainingDims] = dims;
    if (!remainingDims.length) {
      return Array.from({ length: currentDim }, (_, i) => {
        const finalCoords = [...currentCoords, i];
        let flatIndex = 0;
        for (let d = 0; d < finalCoords.length; d++) {
          flatIndex += finalCoords[d] * strides[d];
        }
        return flat[flatIndex];
      });
    }
    return Array.from(
      { length: currentDim },
      (_, i) => build(remainingDims, [...currentCoords, i])
    );
  }
  return build(shape, []);
}

export { reshape };
//# sourceMappingURL=reshape.js.map
//# sourceMappingURL=reshape.js.map