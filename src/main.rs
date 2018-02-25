extern crate gl;
extern crate sdl2;

use std::io::prelude::*;
use std::fs::{File};
use std::env;
use std::sync::{Arc, RwLock};

use sdl2::audio::{AudioCallback, AudioSpecDesired};

mod chip8;

const SCREEN_WIDTH : u32 = 640;
const SCREEN_HEIGHT: u32 = 320;

const FILL_COLOR: (u8, u8, u8, u8) = (255, 252, 135, 17);

fn find_sdl_gl_driver() -> Option<u32> {
	for (index, item) in sdl2::render::drivers().enumerate() {
		if item.name == "opengl" {
			return Some(index as u32);
		}
	}
	None
}


struct MachineSound<'a> { 
    vm: Arc<RwLock<chip8::Chip8<'a>>>,
    volume: f32,
}

impl<'a> AudioCallback for MachineSound<'a> {

    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        let vm = self.vm.read().unwrap();
        let val = if vm.dt > 0u8 { self.volume } else { 0.0 };
        for x in out.iter_mut() {
            *x = val;
        }
    }
}

fn keymap(key: sdl2::keyboard::Keycode) -> usize {

    use sdl2::keyboard::Keycode;

    match key {
        Keycode::Num1 	=> 0x01,
        Keycode::Num2 	=> 0x02,
        Keycode::Num3 	=> 0x03,
        Keycode::Num4 	=> 0x0C,
        Keycode::Q 		=> 0x04,
        Keycode::W 		=> 0x05,
        Keycode::E 		=> 0x06,
        Keycode::R 		=> 0x0D,
        Keycode::A 		=> 0x07,
        Keycode::S 		=> 0x08,
        Keycode::D 		=> 0x09,
        Keycode::F 		=> 0x0E,
        Keycode::Z 		=> 0x0A,
        Keycode::X 		=> 0x00,
        Keycode::C 		=> 0x0B,
        Keycode::V 		=> 0x0F,
        _ => { panic!("not recognized key") },
    }
}

fn main() {

    let mut args = env::args();
    args.next();
    let file = args.next().unwrap_or(String::from("rsrc/games/Pong (alt).ch8"));

    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.push(file);

    let mut file = File::open(current_dir).unwrap();
    let mut program : Vec<u8> = Vec::with_capacity( file.metadata().map( |m| m.len() as usize ).unwrap() );
    file.read_to_end(&mut program).expect("can't read from file");

	let mut memory : [u8 ; 4096] = [0 ; 4096];
	let mut vm = chip8::Chip8::new(&program, &mut memory);
    let vm_p = Arc::new(RwLock::new(vm));


	let sdl_context = sdl2::init().unwrap();


    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(44100),
        channels: Some(1),
        samples: None,
    };

    let device = audio_subsystem.open_playback(None, &desired_spec, |_| {
        let vm = vm_p.clone();
        MachineSound {
            vm,
            volume: 0.4,
        }
    }).unwrap();
    device.resume();



	let video_subsystem = sdl_context.video().unwrap();
    
	let window = video_subsystem.window("Window", SCREEN_WIDTH, SCREEN_HEIGHT)
		.opengl()
		.build()
		.unwrap();

	let mut canvas = window.into_canvas()
		.index(find_sdl_gl_driver().unwrap())
		.build()
		.unwrap();

	let texture_creator = canvas.texture_creator();

	let mut texture = texture_creator.create_texture_streaming(
	    sdl2::pixels::PixelFormatEnum::ARGB8888,
		SCREEN_WIDTH,
		SCREEN_HEIGHT
	).expect("Fail to create texture");



    let mut inputs : [bool; 16] = [false; 16];
	let mut event_poll = sdl_context.event_pump().unwrap();
	'main: loop { 

		for event in event_poll.poll_iter() {
            use sdl2::keyboard::Keycode;	

			match event { 
				sdl2::event::Event::Quit { .. } | 
                sdl2::event::Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
					break 'main;
				},

				sdl2::event::Event::KeyDown { keycode: Some(key), .. } => {
                    let index = keymap(key);
                    inputs[index] = true;
				}

				sdl2::event::Event::KeyUp{ keycode: Some(key), .. } => {
                    let index = keymap(key);
                    inputs[index] = false;
				}
				_ => {}
			};
		}
        
        {
            let mut vm = vm_p.write().unwrap();
            vm.run(&inputs);
        }

		//copy chip8 screen into texture
        // We are manually upscaling from the original resolution (64x32) to final resolution (640x320)
        texture.with_lock(None, |buffer, _pitch| {
            
            let vm = vm_p.read().unwrap();
            for (y_index, scanline) in vm.video_memory.chunks(64).enumerate() {

                //calculate the upscaled scanline start and end indexes
                let start_y = y_index * 10;
                let end_y   = start_y + 10;

                for y in start_y..end_y {
                    for (x_index, pixel) in scanline.iter().enumerate() {

                        //calculate the upscaled x start and end coordinated
                        let start_x = x_index * 10;
                        let end_x = start_x + 10;

                        for x in start_x..end_x {
                            
                            // calculate the pixel start position 
                            let index = (x + (y * 640)) * 4;

                            let colors = [FILL_COLOR.0 * pixel, 
                                          FILL_COLOR.1 * pixel,
                                          FILL_COLOR.2 * pixel,
                                          FILL_COLOR.3 * pixel];
                            
                            //copy the color
                            buffer[index..index+4].clone_from_slice(&colors[..]);
                            
                            /*
                             * This is slower, due to memory access pattern I guess
                             * load memory, make multiplication, store to memory, repeat
                            buffer[index + 0] = FILL_COLOR.0 * pixel;
                            buffer[index + 1] = FILL_COLOR.1 * pixel;
                            buffer[index + 2] = FILL_COLOR.2 * pixel;
                            buffer[index + 3] = FILL_COLOR.3 * pixel;
                            */
                        }
                    }
                }
            }
              
            /*
             * Doesn't like the division, this would heavily reduce performance
             * of rendering (all of this is done for every pixel, every frame
            for (index, pixel) in buffer.chunks_mut(4).enumerate() {
                let x = (index % 640) / 10;
                let y = (index / 640) % 320 / 10;
                let bit_value = vm.video_memory[ x + (y * 64)];
                pixel[0] = FILL_COLOR.0 * bit_value;
                pixel[1] = FILL_COLOR.1 * bit_value;
                pixel[2] = FILL_COLOR.2 * bit_value;
                pixel[3] = FILL_COLOR.3 * bit_value;
            }
            */
        }).expect("Cannot update buffer");

		canvas.copy(&texture, None, None).expect("Can't copy texture into canvas");
		canvas.present();
	}
}
