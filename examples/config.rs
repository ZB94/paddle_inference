use paddle_inference::ctypes::{
    PD_ConfigCreate, PD_ConfigDestroy, PD_ConfigSetModelDir, PD_ConfigSummary,
};

fn main() {
    let config = unsafe { PD_ConfigCreate() };
    let summary = unsafe { PD_ConfigSummary(config) };

    unsafe {
        PD_ConfigSetModelDir(config, b"test\0".as_ptr() as *const _);
    }

    let sa = unsafe { std::slice::from_raw_parts((*summary).data as *const u8, (*summary).size) };
    let s = String::from_utf8_lossy(sa);
    println!("{s}");
    unsafe { PD_ConfigDestroy(config) };
}
