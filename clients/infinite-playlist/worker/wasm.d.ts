// The Rust→WASM core is imported as a CompiledWasm module (see wrangler.toml).
declare module "*.wasm" {
  const mod: WebAssembly.Module;
  export default mod;
}
