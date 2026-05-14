/**
 * Tiny TypeScript fixture for the AST extractor tests.
 *
 * Expected entities:
 *   module:   auth
 *   function: authenticateUser
 *   function: _decodeToken (private — leading underscore)
 *   class:    AuthError       (extends Error)
 *   method:   AuthError.constructor
 *   method:   AuthError.withContext
 *   class:    JWTAuth         (extends AuthBackend)
 *   method:   JWTAuth.constructor
 *   method:   JWTAuth.verify
 */
import * as jwt from 'jsonwebtoken';
import { Errors } from './errors';

export class AuthBackend {
}

export class AuthError extends Error {
    public status: number;
    public context: Record<string, unknown> = {};

    constructor(message: string, status: number = 401) {
        super(message);
        this.status = status;
    }

    withContext(ctx: Record<string, unknown>): AuthError {
        this.context = ctx;
        return this;
    }
}

export class JWTAuth extends AuthBackend {
    private secret: string;

    constructor(secret: string) {
        super();
        this.secret = secret;
    }

    verify(token: string): Record<string, unknown> {
        return jwt.verify(token, this.secret) as Record<string, unknown>;
    }
}

function _decodeToken(token: string, secret: string): Record<string, unknown> {
    return jwt.verify(token, secret) as Record<string, unknown>;
}

export function authenticateUser(token: string, secret: string): Record<string, unknown> {
    const claims = _decodeToken(token, secret);
    const backend = new JWTAuth(secret);
    backend.verify(token);
    return claims;
}
