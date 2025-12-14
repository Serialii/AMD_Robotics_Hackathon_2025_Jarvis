# AMD_Robotics_Hackathon_2025-main
The goal of this project was to develop a practical and helpful robotic companion capable of assisting with everyday tasks. We created Jarvis, a smart robotic arm designed to pick up objects, place them in designated locations, or hand them to a user upon request. Our primary focus was on creating a system that is intuitive, responsive, and reliable, providing meaningful assistance in everyday interactions.

Scope of the project:
1. Interact with our hand by giving and taking objects when we tell him to
2. Tic-Tac-Toe
3. Voice commands

   
## Project Workflow

The development of Jarvis followed a structured workflow, starting with data collection. We designed and collected a **custom dataset** capturing various object manipulation actions performed by the arm in different scenarios. This dataset provided the foundation for training our models, ensuring that Jarvis could generalize across multiple tasks.

Using the dataset, we trained the robotic arm’s behavior with the tools provided by AMD and Hugging Face, leveraging pretrained models and policies to accelerate learning. These tools allowed us to integrate vision and language understanding with action control, enabling Jarvis to interpret both visual inputs and textual commands efficiently.

After training, we combined the models into a modular system, where different specialized policies handle specific actions such as picking, placing, handing objects, greeting the user, or executing game-related moves for Tic Tac Toe. This modular design allows Jarvis to switch between tasks seamlessly.

To make Jarvis easy and natural to interact with, we implemented a voice interface. Spoken commands are converted to text through a speech-to-text system, and the resulting instruction is interpreted to select the appropriate task. High-level commands such as “pick up,” “give me,” “let’s play,” or a simple greeting trigger the corresponding behavior, which is then executed by the robotic arm in real time.

---

## Technical Highlights

The project utilized AMD’s development tools for model training and inference, combined with Hugging Face’s pretrained models and libraries for vision-language processing and speech-to-text functionality. By relying on these tools, we were able to focus on system integration and real-world usability rather than implementing low-level models from scratch.

---

## Conclusion

Jarvis demonstrates how off-the-shelf tools and pretrained models can be combined to create a modular and intelligent robotic assistant. By integrating manipulation, voice interaction, user awareness, and simple gameplay, Jarvis goes beyond a traditional robotic arm to become a more engaging companion. The separation of perception, decision-making, and execution layers ensures that the system is safe, adaptable, and scalable, allowing new tasks and interactions to be added easily. This project represents a meaningful step toward everyday robotic companions that interact naturally with humans.
