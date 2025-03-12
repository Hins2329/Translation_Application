package com.loginAndRegister.loginAndRegister_service.Application;

import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import com.loginAndRegister.loginAndRegister_service.Service.UserService;
import com.loginAndRegister.loginAndRegister_service.Util.JwtUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Component;


@Component
@ComponentScan("com.loginAndRegister.loginAndRegister_service.infrastructure")
public class LRaplication {

    private final UserService userService;
    private final JwtUtil jwtUtil;
    private final BCryptPasswordEncoder passwordEncoder;

//    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    @Autowired
    public LRaplication(UserService userService, JwtUtil jwtUtil, BCryptPasswordEncoder passwordEncoder) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
        this.passwordEncoder = passwordEncoder;
    }

    public String login(User user) {
        try {
            System.out.println("Inside login method");
            User foundUser = userService.findByUsername(user.getUsername());

            if (foundUser == null) {
                System.out.println("User not found");
                throw new RuntimeException("User not found");
            }
            System.out.println("rawPW: "+user.getPassword() );
            System.out.println("Stored Hashed Password: " + foundUser.getPassword());
            System.out.println("Input Raw Password: " + user.getPassword());
            System.out.println(userService.verifyPassword(user.getPassword(), foundUser.getPassword()));
            if(!userService.verifyPassword(user.getPassword(), foundUser.getPassword())) {
                System.out.println("Password does NOT match");
                throw new RuntimeException("Invalid username or password");
            }

            System.out.println("Password MATCHED!");

            return jwtUtil.generateToken(foundUser.getUsername());

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Login error: " + e.getMessage());
        }
    }

    public boolean validateToken(String token){
        return jwtUtil.validateToken(token,jwtUtil.extractUsername(token));
    }


    public void register(User user) {
        userService.registerUser(user);
    }

}
