extern crate rand;

use std::fmt;

#[derive(Debug)]
struct Stack {
	s: [u16 ; 16],
	i: usize,
}

impl Stack {
	fn new() -> Stack {
		Stack {
			s: [0; 16],
			i: 0,
		}
	}

	fn push(&mut self, v: u16) {
		self.s[self.i] = v;
		self.i+=1;

	}

	fn pop(&mut self) -> u16 {
		self.i -= 1;
		self.s[self.i]

	}
}

enum Instruction {

    ClearScreen,                // 00E0 | Clear display
    Return,                     // 00EE | Return from subroutine (Pop stack and jump to address)
    JumpTo(u16),                // 1nnn | Jump to address (set pc to value)
    Call(u16),                  // 2nnn | Call subroutine
    SkipEqualRegVal(u8, u8),    // 3xkk | Skip next instruction if Vx == kk
    SkipNotEqualRegVal(u8, u8), // 4xkk | Skip next instruction if Vx != kk
    SkipEqualRegReg(u8, u8),    // 5xy0 | Skip next instruction if Vx == Vy
    SetRegVal(u8, u8),          // 6xkk | Load value kk into Vx
    AddRegVal(u8, u8),          // 7xkk | Vx = Vx + kk
    SetRegReg(u8, u8),          // 8xy0 | Vx = Vy
    Or(u8, u8),                 // 8xy1 | Vx = Vx OR Vy
    And(u8, u8),                // 8xy2 | Vx = Vx AND Vy
    Xor(u8, u8),                // 8xy3 | Vx = Vx XOR Vy
    AddRegReg(u8, u8),          // 8xy4 | Vx = Vx + Vy (set Vf to one if carry)
    SubRegReg(u8, u8),          // 8xy5 | Vx = Vx - Vy (set vf if NO borrow (Vx > Vy))
    ShiftRight(u8, u8),         // 8xy6 | Vx = Vy >> 1 (Vf set to one if least significant bit of Vy is 1)
    SubNRegReg(u8, u8),         // 8xy7 | Vx = Vy - Vx (set Vr if NO borrow (Vy > Vx))
    ShiftLeft(u8, u8),          // 8xyE | Vx = Vy << 1 (Vf  is set to one if most significant bit is 1)
    SkipNotEqualRegReg(u8, u8), // 9xy0 | Skip next instruction if Vx != Vy
    SetI(u16),                  // Annn | I = nnn
    JumpV0(u16),                // Bnnn | Jump to nnn + V0
    Rnd(u8, u8),                // Cxkk | generate random value between 0..255, the rnd AND kk set to Vx
    Draw(u8, u8, u8),           // Dxyn | Draw the n byte starting at I at the coordinate (Vx, Vy). Drawing is done by XORing each bit, if one bit is set to 0, Vf is set to 1 for collision detection
    SkipKey(u8),                // Ex9E | Skip next instruction is the key coresponding to the value of Vx is pressed
    SkipNotKey(u8),             // ExA1 | Skip next instruction is the key coresponding to the value of Vx is NOT pressed
    ReadDt(u8),                 // Fx07 | Vx = Dt
    WaitKey(u8),                // Fx0A | Vx = K (Stop execution of program, store the value of the first key pressed into Vx)
    SetDT(u8),                  // Fx15 | DT = Vx,
    SetST(u8),                  // Fx18 | ST = Vx,
    AddIReg(u8),                // Fx1E | I = I + Vx
    LoadPreSprite(u8),          // Fx29 | I = Mem location of sprite corresponding to value of Vx 
    SetBCD(u8),                 // Fx33 | Take value of Vx, put the hundred digit into I, tens digit into I+1 and ones digit to i+2
    StoreMem(u8),               // Fx55 | Store value from V0 to Vx in memory starting I
    LoadMem(u8),                // Fx65 | Read value from memory starting at I into V0 to Vx (V0 = I + 0, V1 = I + 1 etc)
    
    NOOP(u8, u8, u8, u8),       // keep original data for debug, it might be sprite data
}

impl fmt::Debug for Instruction {
    
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        
        use self::Instruction::*;
        match *self {
            ClearScreen                      => write!(f, "CLS"),
            Return                           => write!(f, "RET"),
            JumpTo             ( address )   => write!(f, "JMP {:X}", address),
            Call               ( address )   => write!(f, "CALL {:X}", address),
            SkipEqualRegVal    ( r, v )      => write!(f, "SKPE V{:X} {}", r, v),
            SkipNotEqualRegVal ( r, v)       => write!(f, "SKPNE V{:X} {}", r, v),
            SkipEqualRegReg    (r1, r2)      => write!(f, "SKPE V{:X} V{:X}", r1, r2),
            SetRegVal          (r, v)        => write!(f, "SET V{:X} {:X}", r, v ),
            AddRegVal          (r, v)        => write!(f, "ADD V{:X} {}", r, v ),
            SetRegReg          (r1, r2)      => write!(f, "SET V{:X} V{:X}", r1, r2 ),
            Or                 (r1, r2)      => write!(f, "OR V{:X} V{:X}", r1, r2 ),
            And                (r1, r2)      => write!(f, "AND V{:X} V{:X}", r1, r2 ),
            Xor                (r1, r2)      => write!(f, "XOR V{:X} V{:X}", r1, r2 ),
            AddRegReg          (r1, r2)      => write!(f, "ADD V{:X} V{:X}", r1, r2 ),
            SubRegReg          (r1, r2)      => write!(f, "SUB V{:X} V{:X}", r1, r2 ),
            ShiftRight         (r1, r2)      => write!(f, "SHR V{:X} V{:X}", r1, r2 ),
            SubNRegReg         (r1, r2)      => write!(f, "SUBN V{:X} V{:X}", r1, r2 ),
            ShiftLeft          (r1, r2)      => write!(f, "SHL V{:X} V{:X}", r1, r2 ),
            SkipNotEqualRegReg (r1, r2)      => write!(f, "SKPNE V{:X} V{:X}", r1, r2 ),
            SetI               ( address )   => write!(f, "SET I {:X}", address ),
            JumpV0             ( address )   => write!(f, "JMP {:X} V0", address ),
            Rnd                ( r , v )     => write!(f, "RND V{:X} {}", r, v ),
            Draw               ( r1, r2, n ) => write!(f, "DRAW {} V{:X} V{:X}", n, r1, r2 ),
            SkipKey            ( r )         => write!(f, "SKPKey V{:X}", r ),
            SkipNotKey         ( r )         => write!(f, "SKPNKey V{:X}", r ),
            ReadDt             ( r )         => write!(f, "READ V{:X} DT" , r ),
            WaitKey            ( r )         => write!(f, "WAITKey V{}:X", r ),
            SetDT              ( r )         => write!(f, "SET DT V{:X}", r ),
            SetST              ( r )         => write!(f, "SET ST V{:X}", r ),
            AddIReg            ( r )         => write!(f, "ADD I V{:X}", r ),
            LoadPreSprite      ( r )         => write!(f, "LDSPRT V{:X}", r ),
            SetBCD             ( r )         => write!(f, "BCD V{:X}", r ),
            StoreMem           ( r )         => write!(f, "STOREMEM V{:X}", r ),
            LoadMem            ( r )         => write!(f, "LDMEM V{:X}", r ),

            NOOP               (a,b,c,d)     => write!(f, "NO-OP ({:X}{:X} {:X}{:X})", a, b, c, d),
        }
    }
}
	
fn build_value_from(v1: u8, v2: u8) -> u8 {
    (v1 << 4) + v2
}

fn build_address_from(a1: u8, a2: u8, a3: u8) -> u16 {
    ( ( a1 as u16 ) << 8 ) + ( ( a2 as u16 ) << 4 ) + ( a3 as u16 )
}

fn parse_instruction(opcode: u16) -> Instruction {
    
    let tuple_instruction : (u8, u8, u8, u8) = ( 
                                                 ((opcode & 0xF000) >> 12) as u8, 
                                                 ((opcode & 0x0F00) >> 8) as u8,
                                                 ((opcode & 0x00F0) >> 4) as u8, 
                                                 ((opcode & 0x000F) >> 0) as u8 
                                               );
    use self::Instruction::*;

    match tuple_instruction {
        
        (0x0, 0x0, 0xE, 0x0) => ClearScreen,
        (0x0, 0x0, 0xE, 0xE) => Return,
        (0x1,  n1,  n2,  n3) => JumpTo( build_address_from( n1, n2, n3 ) ),
        (0x2,  n1,  n2,  n3) => Call( build_address_from( n1, n2, n3 ) ),
        (0x3,   x,  k1,  k2) => SkipEqualRegVal( x, build_value_from( k1, k2 ) ),
        (0x4,   x,  k1,  k2) => SkipNotEqualRegVal( x, build_value_from( k1, k2 ) ),
        (0x5,   x,   y, 0x0) => SkipEqualRegReg( x, y ),
        (0x6,   x,  k1,  k2) => SetRegVal( x, build_value_from( k1, k2 ) ),
        (0x7,   x,  k1,  k2) => AddRegVal( x, build_value_from( k1, k2 ) ),
        (0x8,   x,   y, 0x0) => SetRegReg( x, y ),
        (0x8,   x,   y, 0x1) => Or( x, y ),
        (0x8,   x,   y, 0x2) => And( x, y ),
        (0x8,   x,   y, 0x3) => Xor( x, y ),
        (0x8,   x,   y, 0x4) => AddRegReg( x, y ),
        (0x8,   x,   y, 0x5) => SubRegReg( x, y ),
        (0x8,   x,   y, 0x6) => ShiftRight( x, y ),
        (0x8,   x,   y, 0x7) => SubNRegReg( x, y ),
        (0x8,   x,   y, 0xE) => ShiftLeft( x, y ),
        (0x9,   x,   y, 0x0) => SkipNotEqualRegReg( x, y ),
        (0xA,  n1,  n2,  n3) => SetI( build_address_from( n1, n2, n3 ) ),
        (0xB,  n1,  n2,  n3) => JumpV0( build_address_from( n1, n2, n3 ) ),
        (0xC,   x,  k1,  k2) => Rnd( x, build_value_from( k1, k2 ) ),
        (0xD,   x,   y,   n) => Draw( x, y, n ),
        (0xE,   x, 0x9, 0xE) => SkipKey( x ),
        (0xE,   x, 0xA, 0x1) => SkipNotKey( x ),
        (0xF,   x, 0x0, 0x7) => ReadDt( x ),
        (0xF,   x, 0x0, 0xA) => WaitKey( x ),
        (0xF,   x, 0x1, 0x5) => SetDT( x ),
        (0xF,   x, 0x1, 0x8) => SetST( x ),
        (0xF,   x, 0x1, 0xE) => AddIReg( x ),
        (0xF,   x, 0x2, 0x9) => LoadPreSprite( x ),
        (0xF,   x, 0x3, 0x3) => SetBCD( x ),
        (0xF,   x, 0x5, 0x5) => StoreMem( x ),
        (0xF,   x, 0x6, 0x5) => LoadMem( x ),

        (a, b ,c ,d) => NOOP(a,b,c,d),
    }

}

pub struct Chip8<'a>{
	pc: u16,			    // program counter
	registers: [u8 ; 16], 	// registers
	i: u16,				    // memory register
	pub dt: u8,				// delay timer
	pub st: u8,				// stack timer
	stack: Stack,		    // program stack
	memory: &'a mut [u8],   // systems memory
    pub video_memory: [u8; 64 * 32],
}

impl<'a> Chip8<'a> {
	pub fn new(program_buffer: &[u8], memory: &'a mut [u8]) -> Chip8<'a> {

        // preloaded sprites
        memory[0x000..0x050].clone_from_slice(
            &[ 
              0xF0, 0x90, 0x90, 0x90, 0xF0, //0
              0x20, 0x60, 0x20, 0x20, 0x70, //1
              0xF0, 0x10, 0xF0, 0x80, 0xF0, //2
              0xF0, 0x10, 0xF0, 0x10, 0xF0, //3
              0x90, 0x90, 0xF0, 0x10, 0x10, //4
              0xF0, 0x80, 0xF0, 0x10, 0xF0, //5
              0xF0, 0x80, 0xF0, 0x90, 0xF0, //6
              0xF0, 0x10, 0x20, 0x40, 0x40, //7
              0xF0, 0x90, 0xF0, 0x90, 0xF0, //8
              0xF0, 0x90, 0xF0, 0x10, 0xF0, //9
              0xF0, 0x90, 0xF0, 0x90, 0x90, //A
              0xE0, 0x90, 0xE0, 0x90, 0xE0, //B
              0xF0, 0x80, 0x80, 0x80, 0xF0, //C
              0xE0, 0x90, 0x90, 0x90, 0xE0, //D
              0xF0, 0x80, 0xF0, 0x80, 0xF0, //E
              0xF0, 0x80, 0xF0, 0x80, 0x80  //F
            ][..]);

        let end_program_address = 0x200 + program_buffer.len();
        memory[0x200..end_program_address].clone_from_slice(program_buffer); 

		Chip8 {
			pc: 0x200,
			registers: [0;16],
			i: 0,
			dt: 0,
			st: 0,
			stack: Stack::new(),
			memory,
            video_memory: [0; 64 * 32 ],
		}
	}

    fn reg(&self, i: u8) ->  u8 {
        self.registers[i as usize]
    }

    fn reg_mut(&mut self, i: u8) -> &mut u8 {
        &mut self.registers[i as usize]
    }

    fn flag_reg_mut(&mut self) -> &mut u8 {
        &mut self.registers[0xF]
    }

    fn inc_pc(&mut self) {
        self.pc += 2;
    }

    pub fn run(&mut self, keyboard_events: &[bool]) {
        
        use self::Instruction::*;

        let opcode = u16_from_big_endian( self.memory[self.pc as usize], self.memory[(self.pc as usize) + 1] );
        let instruction = parse_instruction(opcode);

        match instruction {
            
                ClearScreen                      => { 
                    for i in self.video_memory.iter_mut() { 
                        *i = 0;
                    }; 
                },

                Return                           => { 
                    self.pc = self.stack.pop(); 
                },

                JumpTo             ( address )   => { 
                    self.pc = address; 
                    return; 
                },

                Call               ( address )   => { 
                    self.stack.push(self.pc); 
                    self.pc = address; 
                    return; 
                },

                SkipEqualRegVal    ( r, v )      => { 
                    if self.reg(r) == v { 
                        self.inc_pc(); 
                    } 
                },

                SkipNotEqualRegVal ( r, v )       => { 
                    if self.reg(r) != v { 
                        self.inc_pc(); 
                    } 
                },

                SkipEqualRegReg    (r1, r2)      => { 
                    if self.reg(r1) == self.reg(r2) { 
                        self.inc_pc(); 
                    } 
                },

                SetRegVal          ( r, v )        => { 
                    *self.reg_mut(r)  = v; 
                },

                AddRegVal          ( r, v )        => { 

                    use std::num::Wrapping;

                    let a = Wrapping(self.reg(r));
                    let b = Wrapping(v);

                    *self.reg_mut(r) = (a + b).0; 
                },

                SetRegReg          ( r1, r2 )      => { 
                    *self.reg_mut(r1) = self.reg(r2); 
                },

                Or                 ( r1, r2 )      => { 
                    *self.reg_mut(r1) |= self.reg(r2); 
                },

                And                ( r1, r2 )      => { 
                    *self.reg_mut(r1) &= self.reg(r2); 
                },

                Xor                ( r1, r2 )      => { 
                    *self.reg_mut(r1) ^= self.reg(r2); 
                },

                AddRegReg          ( r1, r2 )      => { 

                    let (a, b) = (self.reg(r1), self.reg(r2));

                    let (res, overflow) = a.overflowing_add(b);
                    *self.reg_mut(r1) = res; 
                    *self.flag_reg_mut() = if overflow { 1 } else { 0 };
                },

                SubRegReg          ( r1, r2 )      => { 

                    let (a, b) = (self.reg(r1), self.reg(r2));
                    
                    let (res, overflow) = a.overflowing_sub(b);
                    *self.reg_mut(r1) = res; 
                    *self.flag_reg_mut() = if overflow{ 0 } else { 1 };
                },

                ShiftRight         ( r1, r2 )      => { 

                    let (a, b) = (self.reg(r1), self.reg(r2));

                    let (res, overflow) = a.overflowing_shr(b as u32);
                    *self.flag_reg_mut() = if overflow { 1 } else { 0 };
                    *self.reg_mut(r1) = res; 
                },

                SubNRegReg         ( r1, r2 )      => { 

                    let (a, b) = (self.reg(r1), self.reg(r2));
                    
                    let (res, overflow) = b.overflowing_sub(a);
                    *self.reg_mut(r1) = res; 
                    *self.flag_reg_mut() = if overflow { 0 } else { 1 };
                },

                ShiftLeft          ( r1, r2 )      => { 

                    let (a, b) = (self.reg(r1), self.reg(r2));

                    let (res, overflow) = a.overflowing_shl(b as u32);
                    *self.reg_mut(r1) = res; 
                    *self.flag_reg_mut() = if overflow { 1 } else { 0 };
                },

                SkipNotEqualRegReg ( r1, r2 )      => { 
                    if self.reg(r1) != self.reg(r2) { 
                        self.inc_pc(); 
                    } 
                },

                SetI               ( address )   => { 
                    self.i = address; 
                },

                JumpV0             ( address )   => { 
                    self.pc = address + (self.reg(0) as u16); 
                    return; 
                },

                Rnd                ( r , v )     => {
                    let rnd_num : u8 = rand::random::<u8>();
                    *self.reg_mut(r) = rnd_num & v;
                },

                Draw               ( r1, r2, n ) => { 

                    *self.flag_reg_mut() = 0;

                    let (x, y) = (self.reg(r1), self.reg(r2));
                    
                    for (y_index, current_y) in (y..y+n).enumerate() {
                        
                        let y = current_y % 32; // needs to wrap around
                        
                        let memory_index = (self.i as usize) + y_index;
                        let byte_to_copy = self.memory[memory_index];
                        
                        for (x_index, current_x) in (x..x+8).enumerate() {

                            let pixel_value = byte_to_copy & (0x80 >> x_index); //get the bit at x_index
                            if pixel_value > 0 {

                                let pixel_value = pixel_value >> (7 - x_index);

                                let x = current_x % 64; //needs to wrap around
                                let pixel_index =  ((y as usize) * 64) + x as usize;
                                let pixel = self.video_memory[pixel_index];

                                if pixel > 0 {
                                    *self.flag_reg_mut() = 1;
                                }
                                let pixel = &mut self.video_memory[pixel_index];
                                *pixel = *pixel ^ pixel_value;
                            }
                        }
                    }
                },

                SkipKey            ( r )         => { 
                    if keyboard_events[self.reg(r) as usize] { 
                        self.inc_pc();
                    } 
                },

                SkipNotKey         ( r )         => { 
                    if !keyboard_events[self.reg(r) as usize] { 
                        self.inc_pc();
                    } 
                },

                ReadDt             ( r )         => {
                    *self.reg_mut(r) = self.dt;
                },

                WaitKey            ( r )         => {
                    
                    let mut key_pressed = keyboard_events.iter().enumerate().filter( |&(_, &v)| v );
                    match key_pressed.next() {
                        Some((index, _)) => *self.reg_mut(r) = index as u8,
                        None => return,
                    };
                },

                SetDT              ( r )         => {
                    self.dt = self.reg(r);
                },

                SetST              ( r )         => {
                    self.st = self.reg(r);
                },

                AddIReg            ( r )         => {
                    self.i += self.reg(r) as u16;
                },

                LoadPreSprite      ( r )         => {
                    let val = self.reg(r) as u16; 
                    self.i = val * 5;
                },

                SetBCD             ( r )         => {
                    let val = self.reg(r);

                    let cent = val / 100;
                    let dec = (val - cent) / 10;
                    let unit = val - cent - dec;

                    let i = self.i as usize;

                    self.memory[i + 0] = cent;
                    self.memory[i + 1] = dec;
                    self.memory[i + 2] = unit;
                },

                StoreMem           ( r )         => {
                    for i in 0..r {
                        self.memory[(self.i as usize) + i as usize] = self.reg(i);
                    }
                },

                LoadMem            ( r )         => {
                    for i in 0..r {
                        *self.reg_mut(i) = self.memory[(self.i as usize) + i as usize];
                    }
                },

                NOOP(_, _, _, _) => panic!("Should never happen"),
        }

        self.inc_pc();
        self.dt = self.dt.saturating_sub(1);
        self.st = self.dt.saturating_sub(1);
    }
}

impl<'a> fmt::Display for Chip8<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Memory:\n{:?}\n\nProgram Counter: {}\nRegisters: {:?}\nI: {}\nDT: {}\nST: {}\n",
                  self.memory,
                  self.pc,
                  self.registers,
                  self.i,
                  self.dt,
                  self.st
              )
    }
}

fn u16_from_big_endian(high: u8, low: u8) -> u16 {
    ((high as u16) << 8) + low as u16
}

mod test {
	use super::*;

    #[test]
    fn test_that_overflowing_sets_vf_properly() {

        let mut memory = [0u8 ; 4096];
        let program = [0x80u8, 0x14u8,  //add r0, r1
                       0x80u8, 0x15u8,  //sub with overflow, r0, r1
                       0x80u8, 0x17u8]; //sub not overflow r0, r1

        let mut vm = Chip8::new(&program[..], &mut memory[..]);

        *vm.reg_mut(0x0) = 255;
        *vm.reg_mut(0x1) = 100;
        
        run(&mut vm, &[][..]);

        assert_eq!(vm.flag_reg(), 1);
        assert_eq!(vm.reg(0x0), 99);


        *vm.reg_mut(0x0) = 0;
        *vm.reg_mut(0x1) = 1;

        run(&mut vm, &[][..]);

        assert_eq!(vm.flag_reg(), 0);
        assert_eq!(vm.reg(0x0), 255);



        *vm.reg_mut(0x0) = 1;
        *vm.reg_mut(0x1) = 0;

        run(&mut vm, &[][..]);

        assert_eq!(vm.flag_reg(), 0);
        assert_eq!(vm.reg(0x0), 255);
    }
}
