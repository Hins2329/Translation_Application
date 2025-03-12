package com.loginAndRegister.loginAndRegister_service.Controller;
import com.loginAndRegister.loginAndRegister_service.Application.LRaplication;
import com.loginAndRegister.loginAndRegister_service.Application.RegisterApplication;
import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import com.loginAndRegister.loginAndRegister_service.Service.UserService;
import com.loginAndRegister.loginAndRegister_service.infrastructure.MybatisRepository.UserRepositoryImpl;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/users")
//@CrossOrigin(origins = "*") // Allow requests from different origins
public class UserController {

    @Autowired
    private final LRaplication lraplication;
    private final RegisterApplication registerApplication;
    private final UserService userService ;
    private final UserRepositoryImpl userRepository;


    public UserController(LRaplication lraplication, RegisterApplication registerApplication, UserService userService, UserRepositoryImpl userRepository) {
        this.lraplication = lraplication;
        this.registerApplication = registerApplication;
        this.userService = userService;
        this.userRepository = userRepository;
    }
//    @PostMapping("/test")
//    public void test(@RequestBody User user){
//        User foundUser = userRepository.findByUsername("1234");
//        System.out.println("found");
//
//    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody User user, HttpServletResponse response){
        try{
            String token = lraplication.login(user);
            System.out.println("Generated JWT: " + token);

            ResponseCookie cookie = ResponseCookie.from("token", token)
                    .httpOnly(true) // Prevent JS from accessing it
                    .secure(false) // 🔥 Set to false for local dev, true for production
                    .sameSite("None") // ✅ For cross-origin requests
                    .path("/")
                    .maxAge(Duration.ofHours(1)) // Expire in 1 hour
                    .build();
            response.addHeader("Set-Cookie", cookie.toString());

            Map<String, String> rs = new HashMap<>();
            rs.put("message", "Login successful");
            rs.put("token", token);
            return ResponseEntity.ok(rs);
        } catch (Exception e) {
//            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Invalid username or password");
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Invalid username or password");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse); // ✅ Return JSON
        }
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody User user){
        System.out.println("im in register");
        try{
            user.setRole("U");
            lraplication.register(user);
            Map<String, String> response = new HashMap<>();
            response.put("message", "Registration successful");
            return ResponseEntity.ok(response); // ✅ Return JSON
        }catch (Exception e){
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Registration failed: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse); // ✅ Return JSON
        }

    }






//    @PostMapping("/save")
//    public void saveUser(@RequestBody User user) {
//        userService.saveUser(user);
//    }
//
//    @GetMapping("/find")
//    public int getUser(@RequestParam User user) {
//        return userService.checkUser(user);
////        if(userService.checkUser(user)==null){
////            return 0;
////        }
////        return 1;
//    }

}
