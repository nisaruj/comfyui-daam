// src/index.ts
var StringFromCodePoint = class extends String {
  constructor(buf, byteOffset, length) {
    const uint32 = new Uint32Array(buf, byteOffset, length);
    const number_arr = Array.from(uint32);
    const str = String.fromCodePoint(...number_arr);
    super(str);
  }
};
var textDecoder = new TextDecoder("latin1");
function readHeader(buf) {
  const view = new DataView(buf);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
    view.getUint8(4),
    view.getUint8(5)
  );
  if (magic !== "\x93NUMPY") throw new Error("Not an .npy file");
  const major = view.getUint8(6);
  const minor = view.getUint8(7);
  let headerLen;
  if (major <= 1) {
    headerLen = view.getUint16(8, true);
    return { headerOffset: 10, headerLen, version: [major, minor] };
  } else {
    headerLen = Number(view.getUint32(8, true));
    return { headerOffset: 12, headerLen, version: [major, minor] };
  }
}
function parseDict(dictStr) {
  const dtype = /'descr'\s*:\s*'([^']+)'/.exec(dictStr)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(dictStr)?.[1] === "True";
  const shapeMatch = /'shape'\s*:\s*\(([^)]*)\)/.exec(dictStr)?.[1] ?? "";
  const shape = shapeMatch.split(",").map((s) => s.trim()).filter(Boolean).map((n) => Number(n));
  if (shape.length === 1 && dictStr.includes("(n,)")) ;
  return { dtype, fortranOrder: fortran, shape };
}
function dtypeToArray(dtype, buf, offset, opts) {
  const little = dtype.startsWith("<") || dtype.startsWith("|");
  const code = dtype.substring(dtype.length - 2);
  if (code[0] === "U") {
    const size = parseInt(code.substring(1));
    const _string = String(new StringFromCodePoint(buf, offset));
    const strings = [];
    for (let i = 0; i < _string.length; i += size) {
      strings.push(_string.substring(i, i + size).replace(/\0/g, ""));
    }
    return strings;
  }
  switch (code) {
    case "b1": {
      if (little) {
        const u8 = new Uint8Array(buf, offset);
        const bools = new Array(u8.length);
        for (let i = 0; i < u8.length; i++) bools[i] = u8[i] !== 0;
        return bools;
      } else {
        return new Uint8Array(buf, offset);
      }
    }
    case "i1":
      return new Int8Array(buf, offset);
    case "u1":
      return new Uint8Array(buf, offset);
    case "i2":
      return new Int16Array(buf, offset);
    case "u2":
      return new Uint16Array(buf, offset);
    case "i4":
      return new Int32Array(buf, offset);
    case "u4":
      return new Uint32Array(buf, offset);
    case "i8":
      return new BigInt64Array(buf, offset);
    case "u8":
      return new BigUint64Array(buf, offset);
    case "f4":
      return new Float32Array(buf, offset);
    case "f8":
      return new Float64Array(buf, offset);
    case "f2": {
      if (opts.convertFloat16 !== false) {
        const u16 = new Uint16Array(buf, offset);
        const f32 = new Float32Array(u16.length);
        for (let i = 0; i < u16.length; i++) f32[i] = f16toF32(u16[i]);
        return f32;
      }
      return new Uint16Array(buf, offset);
    }
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}
function f16toF32(u16) {
  const s = (u16 & 32768) >> 15;
  const e = (u16 & 31744) >> 10;
  const f = u16 & 1023;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  if (e === 31) return f ? NaN : s ? -Infinity : Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}
async function load(source, opts = {}) {
  let buf;
  if (typeof source === "string") {
    const res = await fetch(source);
    buf = await res.arrayBuffer();
  } else if (source instanceof ArrayBuffer) {
    buf = source;
  } else if (source instanceof Blob) {
    buf = await source.arrayBuffer();
  } else {
    buf = source.buffer;
  }
  const { headerOffset, headerLen } = readHeader(buf);
  const headerBytes = new Uint8Array(buf, headerOffset, headerLen);
  const header = textDecoder.decode(headerBytes).trim();
  const { dtype, fortranOrder, shape } = parseDict(header);
  if (!dtype || !shape) throw new Error("Malformed .npy header");
  const dataOffset = headerOffset + headerLen;
  const data = dtypeToArray(dtype, buf, dataOffset, opts);
  return { data, shape, dtype: dtype.slice(1), fortranOrder };
}
function float16ToFloat32(u16) {
  return f16toF32(u16);
}
function arrayToDtype(array) {
  if (array instanceof Uint8Array) {
    return "u1";
  }
  if (array instanceof Uint8ClampedArray) {
    return "u1";
  }
  if (array instanceof Int8Array) {
    return "i1";
  }
  if (array instanceof Uint16Array) {
    return "u2";
  }
  if (array instanceof Int16Array) {
    return "i2";
  }
  if (array instanceof Uint32Array) {
    return "u4";
  }
  if (array instanceof Int32Array) {
    return "i4";
  }
  if (array instanceof Float32Array) {
    return "f4";
  }
  if (array instanceof BigUint64Array) {
    return "u8";
  }
  if (array instanceof BigInt64Array) {
    return "i8";
  }
  if (array instanceof Float64Array) {
    return "f8";
  }
  const kind = typeof array === "object" ? array?.constructor?.name : typeof array;
  throw new TypeError(`Unsupported dtype for ${kind}`);
}
function arrayToTypedArray(dtype, array) {
  if (!Array.isArray(array)) throw new TypeError("Expected an array");
  if (dtype.startsWith("U")) {
    const size = parseInt(dtype.substring(1));
    const buf = new ArrayBuffer(array.length * size * 4);
    const uint32 = new Uint32Array(buf);
    for (let i = 0; i < array.length; i++) {
      const str = array[i];
      for (let j = 0; j < size; j++) {
        const code = j < str.length ? str.codePointAt(j) ?? 0 : 0;
        uint32[i * size + j] = code;
      }
    }
    return new Uint8Array(buf);
  }
  switch (dtype) {
    case "b1":
      return new Uint8Array(array);
    case "i1":
      return new Int8Array(array);
    case "u1":
      return new Uint8Array(array);
    case "i2":
      return new Int16Array(array);
    case "u2":
      return new Uint16Array(array);
    case "i4":
      return new Int32Array(array);
    case "u4":
      return new Uint32Array(array);
    case "i8":
      return new BigInt64Array(array);
    case "u8":
      return new BigUint64Array(array);
    case "f4":
      return new Float32Array(array);
    case "f8":
      return new Float64Array(array);
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}
function inferUnicodeDtypeFromStringArray(array) {
  let longestStringLength = array[0].length;
  for (let i = 1; i < array.length; i++) {
    const element = array[i];
    if (typeof element === "string" && element.length > longestStringLength) {
      longestStringLength = element.length;
    }
  }
  return `U${Math.max(1, longestStringLength)}`;
}
function inferDtypeFromNumberArray(array) {
  let isInteger = true;
  let isNonNegative = true;
  let maxAbsValue = 0;
  for (const num of array) {
    if (!Number.isInteger(num)) {
      isInteger = false;
    }
    if (num < 0) {
      isNonNegative = false;
    }
    const absNum = Math.abs(num);
    if (absNum > maxAbsValue) {
      maxAbsValue = absNum;
    }
  }
  if (!isInteger) {
    if (maxAbsValue <= 340282347e30) return "f4";
    return "f8";
  }
  if (isNonNegative) {
    if (maxAbsValue <= 255) return "u1";
    if (maxAbsValue <= 65535) return "u2";
    if (maxAbsValue <= 4294967295) return "u4";
    return "u8";
  } else {
    if (maxAbsValue <= 127) return "i1";
    if (maxAbsValue <= 32767) return "i2";
    if (maxAbsValue <= 2147483647) return "i4";
    return "i8";
  }
}
function inferDtypeFromArray(array) {
  if (array.length === 0) return "f8";
  const first = array[0];
  if (typeof first === "number") {
    return inferDtypeFromNumberArray(array);
  }
  if (typeof first === "string") {
    return inferUnicodeDtypeFromStringArray(array);
  }
  if (Array.isArray(first)) {
    return inferDtypeFromArray(first);
  }
  if (typeof first === "boolean") {
    return "b1";
  }
  throw new TypeError("Array elements must be numbers, strings or booleans");
}
function isLittleEndian() {
  return new Uint32Array(new Uint8Array([1, 0, 0, 0]).buffer)[0] === 1;
}
function createPyDescription(dtype, shape) {
  const isByte = dtype == "u1" || dtype == "i1";
  const endianness = isByte ? "|" : isLittleEndian() ? "<" : ">";
  const descr = `${endianness}${dtype}`;
  let pyShape = shape.map((v) => {
    return `${v}`;
  }).join(",");
  if (shape.length === 1) pyShape += ",";
  return `{'descr':'${descr}','fortran_order':False,'shape':(${pyShape})}`;
}
function dump(array, shape) {
  const dtype = array instanceof Array ? inferDtypeFromArray(array) : arrayToDtype(array);
  array = array instanceof Array ? arrayToTypedArray(dtype, array) : array;
  let pyDesc = createPyDescription(dtype, shape ?? [array.length]);
  let headerSize = 10 + pyDesc.length;
  const pad = 8 - (headerSize + 1) % 8;
  pyDesc = pyDesc + " ".repeat(pad) + "\n";
  headerSize += pad + 1;
  const buffer = new ArrayBuffer(headerSize + array.byteLength);
  const view = new DataView(buffer);
  view.setUint32(0, 2471384397, false);
  view.setUint32(4, 1348010240, false);
  view.setUint16(8, pyDesc.length, true);
  const encoder = new TextEncoder();
  const header = new Uint8Array(buffer, 10, pyDesc.length);
  encoder.encodeInto(pyDesc, header);
  const data = new Uint8Array(buffer, 10 + pyDesc.length);
  data.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
  return buffer;
}
var N = class {
  constructor(opts = {}) {
    this.opts = opts;
  }
  async load(source) {
    return load(source, this.opts);
  }
  static float16ToFloat32(u16) {
    return f16toF32(u16);
  }
  dump(array, shape) {
    return dump(array, shape);
  }
};

export { arrayToTypedArray, N as default, dump, float16ToFloat32, inferDtypeFromArray, load };
//# sourceMappingURL=index.js.map
//# sourceMappingURL=index.js.map