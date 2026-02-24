/// Implementação da Regra de Hebb para Portas Lógicas
/// Regra: Δw = x * y (onde x é entrada e y é saída desejada)

#[derive(Debug, Clone)]
struct HebbNetwork {
    w1: f32,
    w2: f32,
    bias: f32,
}

impl HebbNetwork {
    fn new() -> Self {
        Self {
            w1: 0.0,
            w2: 0.0,
            bias: 0.0,
        }
    }

    fn train(&mut self, inputs: &[[i32; 2]], targets: &[i32]) {
        for (i, (input, &target)) in inputs.iter().zip(targets.iter()).enumerate() {
            let x1 = input[0];
            let x2 = input[1];
            let y = target;

            self.w1 += (x1 * y) as f32;
            self.w2 += (x2 * y) as f32;
            self.bias += y as f32;

            println!(
                "Amostra {}: Pesos atualizados -> W1: {:.1}, W2: {:.1}, Bias: {:.1}",
                i + 1,
                self.w1,
                self.w2,
                self.bias
            );
        }
    }

    fn predict(&self, x1: i32, x2: i32) -> i32 {
        let soma = (x1 as f32 * self.w1) + (x2 as f32 * self.w2) + self.bias;
        if soma >= 0.0 { 1 } else { -1 }
    }

    fn test(&self, inputs: &[[i32; 2]], targets: &[i32]) {
        println!("\n--- Teste Final ---");
        println!("{:-<50}", "");

        let mut acertos = 0;
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let predicao = self.predict(input[0], input[1]);
            let status = if predicao == target { "✓" } else { "✗" };

            println!(
                "Entrada: [{:2}, {:2}] | Alvo: {:2} | Predição: {:2} {}",
                input[0], input[1], target, predicao, status
            );

            if predicao == target {
                acertos += 1;
            }
        }

        println!("{:-<50}", "");
        println!(
            "Acurácia: {}/{} ({:.0}%)",
            acertos,
            inputs.len(),
            (acertos as f32 / inputs.len() as f32) * 100.0
        );
    }
}

struct LogicGate {
    name: &'static str,
    description: &'static str,
    inputs: [[i32; 2]; 4],
    targets: [i32; 4],
}

fn main() {
    let gates = [
        LogicGate {
            name: "AND",
            description: "Retorna 1 apenas quando ambas entradas são 1",
            inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            targets: [1, -1, -1, -1],
        },
        LogicGate {
            name: "OR",
            description: "Retorna 1 quando pelo menos uma entrada é 1",
            inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            targets: [1, 1, 1, -1],
        },
        LogicGate {
            name: "NAND",
            description: "Negação do AND - retorna -1 apenas quando ambas são 1",
            inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            targets: [-1, 1, 1, 1],
        },
        LogicGate {
            name: "NOR",
            description: "Negação do OR - retorna 1 apenas quando ambas são -1",
            inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            targets: [-1, -1, -1, 1],
        },
    ];

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     REGRA DE HEBB - APRENDIZADO DE PORTAS LÓGICAS           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for gate in &gates {
        println!("\n{:=^60}", "");
        println!("  PORTA: {} ", gate.name);
        println!("  {}", gate.description);
        println!("{:=^60}\n", "");

        println!("Tabela Verdade (bipolar):");
        println!("  X1  | X2  | Saída");
        println!("  ----|-----|------");
        for (input, &target) in gate.inputs.iter().zip(gate.targets.iter()) {
            println!("  {:3} | {:3} | {:3}", input[0], input[1], target);
        }
        println!();

        let mut network = HebbNetwork::new();

        println!("--- Iniciando Treinamento (Regra de Hebb) ---");
        network.train(&gate.inputs, &gate.targets);

        println!(
            "\nPesos Finais: W1={:.1}, W2={:.1}, Bias={:.1}",
            network.w1, network.w2, network.bias
        );

        network.test(&gate.inputs, &gate.targets);
    }
}